"""Streamlit-продукт (Фаза G1): режим «Антифрод / AML».

Загружает обученную edge-GNN (чекпоинт), скорит сэмпл транзакций IBM AML test,
показывает таблицу подозрительных и визуализирует ego-подграф вокруг выбранной
транзакции с подсветкой подозрительных рёбер и найденной цепочки (паттерна).

Запуск:
    streamlit run app/streamlit_app.py

Тяжёлые функции (load/score/figure) вынесены отдельно и тестируемы без Streamlit;
UI собирается в main(). Чекпоинт-агностично (base GINe / Multi-GNN) — путь в сайдбаре.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CKPT_CHOICES = {
    "GINe base (лучший GNN, AUC-PR 0.054)": "checkpoints/ibm_gine_fulldata.pt",
    "Multi-GNN full (метод КР)": "checkpoints/ibm_multignn_fulldata.pt",
}


def load_data_and_scores(ckpt_path: str, max_rows, n_neg: int):
    """Загрузить IBM AML, модель и проскорить сэмпл test-рёбер. → (data, df, ckpt)."""
    import pandas as pd

    from src.datasets import load_ibm_aml
    from src.eval import load_edge_model, sample_test_edges, score_edges

    data, _ = load_ibm_aml(max_rows=max_rows, include_time=False)
    model, ckpt = load_edge_model(ckpt_path)
    idx = sample_test_edges(data, n_neg=n_neg)
    scores = score_edges(model, ckpt, data, idx, batch_size=1024)

    src = data.edge_index[0].numpy()
    dst = data.edge_index[1].numpy()
    amt = np.expm1(data.edge_attr[:, 0].numpy())
    y = data.edge_label.numpy()
    df = pd.DataFrame({
        "edge": idx, "from": src[idx], "to": dst[idx],
        "amount": np.round(amt[idx], 2), "score": np.round(scores, 4),
        "is_laundering": y[idx], "pattern": data.edge_pattern[idx],
    }).sort_values("score", ascending=False).reset_index(drop=True)
    return data, df, ckpt


def build_figure(sub: dict, threshold: float):
    """Plotly-граф ego-подграфа: центр — красный, score≥threshold — оранжевый, фон — серый."""
    import networkx as nx
    import plotly.graph_objects as go

    g = nx.Graph()
    g.add_nodes_from(sub["nodes"])
    for e in sub["edges"]:
        g.add_edge(e["u"], e["v"])
    pos = nx.spring_layout(g, seed=42, k=0.6)

    edge_traces = []
    for e in sub["edges"]:
        x0, y0 = pos[e["u"]]
        x1, y1 = pos[e["v"]]
        if e["is_center"]:
            color, width = "#d62728", 4
        elif e["score"] is not None and e["score"] >= threshold:
            color, width = "#ff7f0e", 2.5
        else:
            color, width = "#b0b0b0", 1
        hover = (f"{e['u']}→{e['v']}<br>сумма: {e['amount']}<br>"
                 f"score: {e['score']}<br>паттерн: {e['pattern']}<br>illicit: {e['label']}")
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None], mode="lines",
            line=dict(color=color, width=width), hoverinfo="text", text=hover, showlegend=False))

    nx_, ny_ = zip(*[pos[n] for n in sub["nodes"]]) if sub["nodes"] else ([], [])
    center_nodes = set(sub["center"])
    node_trace = go.Scatter(
        x=list(nx_), y=list(ny_), mode="markers",
        marker=dict(size=[16 if n in center_nodes else 9 for n in sub["nodes"]],
                    color=["#d62728" if n in center_nodes else "#1f77b4" for n in sub["nodes"]],
                    line=dict(width=1, color="white")),
        text=[f"счёт {n}" for n in sub["nodes"]], hoverinfo="text", showlegend=False)

    fig = go.Figure(edge_traces + [node_trace])
    fig.update_layout(
        title="Ego-подграф (красное — выбранная транзакция, оранжевое — подозрительные)",
        showlegend=False, margin=dict(l=10, r=10, t=40, b=10), height=520,
        xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def main() -> None:
    import streamlit as st

    from src.graph_build import detect_chain, ego_subgraph

    st.set_page_config(page_title="AML Антифрод — GNN", layout="wide")
    st.title("🕵️ Антифрод / AML — детекция подозрительных транзакций (edge-GNN)")
    st.caption("IBM AML HI-Small · направленный мультиграф «счёт→транзакция» · "
               "edge-classification. Модель скорит транзакции; граф подсвечивает цепочку.")

    with st.sidebar:
        st.header("Настройки")
        ckpt_label = st.selectbox("Модель (чекпоинт)", list(CKPT_CHOICES))
        ckpt_path = CKPT_CHOICES[ckpt_label]
        max_rows = st.select_slider(
            "Объём данных (строк CSV)", options=[200_000, 400_000, 1_000_000, None],
            value=400_000, format_func=lambda v: "весь датасет" if v is None else f"{v:,}")
        n_neg = st.slider("Фон легитимных в сэмпле", 500, 5000, 2000, step=500)
        num_hops = st.slider("Глубина ego-подграфа (hops)", 1, 2, 1)

    if not os.path.exists(ckpt_path):
        st.error(f"Чекпоинт не найден: {ckpt_path}. См. CLAUDE.md — перенос с ПК.")
        st.stop()

    cached = st.cache_data(show_spinner="Загрузка данных и скоринг…")(load_data_and_scores)
    data, df, ckpt = cached(ckpt_path, max_rows, n_neg)
    thr = st.sidebar.slider("Порог подозрительности", 0.0, 1.0, float(ckpt["threshold"]), 0.01)

    c1, c2 = st.columns(2)
    c1.metric("Транзакций в сэмпле", len(df))
    c2.metric("Помечено подозрительными (≥порог)", int((df["score"] >= thr).sum()))

    st.subheader("Подозрительные транзакции (по убыванию score)")
    show = df.head(50).copy()
    st.dataframe(show, use_container_width=True, height=320)

    st.subheader("Разбор цепочки")
    pick = st.selectbox(
        "Выберите транзакцию (строка таблицы)", show.index,
        format_func=lambda i: f"#{i}: {show.loc[i,'from']}→{show.loc[i,'to']} (score {show.loc[i,'score']})")
    center = int(show.loc[pick, "edge"])
    info = detect_chain(data, center)
    cols = st.columns(4)
    cols[0].metric("score", f"{show.loc[pick,'score']:.3f}")
    cols[1].metric("fan-out отправителя", info["fan_out"])
    cols[2].metric("fan-in получателя", info["fan_in"])
    cols[3].metric("реципрокность", "да" if info["reciprocal"] else "нет")
    st.write(f"**Истинная метка:** {'illicit' if info['label'] else 'licit'} · "
             f"**размеченный паттерн:** `{info['pattern']}`")

    sub = ego_subgraph(data, center, scores=df["score"].to_numpy(),
                       score_idx=df["edge"].to_numpy(), num_hops=num_hops)
    st.plotly_chart(build_figure(sub, thr), use_container_width=True)
    st.caption(f"Узлов: {len(sub['nodes'])}, рёбер: {len(sub['edges'])}. "
               "Наведите курсор на ребро — сумма, score, паттерн.")


if __name__ == "__main__":
    main()
