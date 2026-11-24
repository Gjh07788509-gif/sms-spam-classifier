"""Train with python app.py --train. Open the demo with streamlit run app.py."""

import hashlib
import io
import json
import platform
import re
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

import joblib
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "sms.zip"
OUTPUT = ROOT / "outputs"
SOURCE = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
SEED = 42
NORMAL_SAMPLE = "Are we still meeting for lunch tomorrow?"
SPAM_SAMPLE = "URGENT! You have won a cash prize. Call now to claim your reward."


def load_data():
    """Validate the archive and remove duplicated or conflicting messages."""
    if DATA.exists():
        payload = DATA.read_bytes()
    else:
        with urlopen(SOURCE, timeout=25) as response:
            payload = response.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        raw = archive.read("SMSSpamCollection").decode("utf-8-sig")

    rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        label, separator, message = line.partition("\t")
        if not separator or label not in {"ham", "spam"}:
            raise ValueError("数据行必须包含 ham 或 spam 标签和制表符。")
        rows.append((label, message))

    frame = pd.DataFrame(rows, columns=["label", "text"])
    frame["key"] = frame["text"].map(lambda text: re.sub(r"\s+", " ", text).strip().lower())
    frame = frame[frame["key"].ne("")].copy()
    label_counts = frame.groupby("key")["label"].transform("nunique")
    frame = frame[label_counts.eq(1)].drop_duplicates("key").reset_index(drop=True)
    frame["target"] = frame["label"].map({"ham": 0, "spam": 1})

    if not DATA.exists():
        DATA.parent.mkdir(exist_ok=True)
        DATA.write_bytes(payload)
    return frame, len(rows), hashlib.sha256(raw.encode("utf-8")).hexdigest()


def evaluate(model, frame):
    target = frame["target"]
    score = model.predict_proba(frame["text"])[:, 1]
    predicted = model.predict(frame["text"])
    matrix = confusion_matrix(target, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    metrics = {
        "macro_f1": float(f1_score(target, predicted, average="macro")),
        "spam_precision": float(precision_score(target, predicted, zero_division=0)),
        "spam_recall": float(recall_score(target, predicted, zero_division=0)),
        "ham_false_positive_rate": float(fp / (tn + fp)),
        "average_precision": float(average_precision_score(target, score)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    return metrics, predicted, score


def train():
    frame, original_count, data_hash = load_data()
    development, test = train_test_split(
        frame, test_size=0.2, stratify=frame["target"], random_state=SEED
    )
    training, validation = train_test_split(
        development, test_size=0.25, stratify=development["target"], random_state=SEED
    )

    classifiers = {
        "MultinomialNB": MultinomialNB(alpha=0.5),
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEED
        ),
    }
    models = {}
    validation_results = {}
    for name, classifier in classifiers.items():
        model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            classifier,
        )
        model.fit(training["text"], training["target"])
        models[name] = model
        validation_results[name] = evaluate(model, validation)[0]

    winner = max(validation_results, key=lambda name: validation_results[name]["macro_f1"])
    model = clone(models[winner]).fit(development["text"], development["target"])
    test_metrics, predicted, score = evaluate(model, test)

    OUTPUT.mkdir(exist_ok=True)
    splits = {"train": training, "validation": validation, "test": test}
    for name, part in splits.items():
        part[["label", "text"]].to_csv(
            OUTPUT / f"{name}.csv", index=False, encoding="utf-8-sig"
        )

    predictions = test[["label", "text"]].copy()
    predictions["predicted_label"] = ["spam" if label else "ham" for label in predicted]
    predictions["spam_score"] = score
    predictions.to_csv(OUTPUT / "test_predictions.csv", index=False, encoding="utf-8-sig")
    errors = predictions[predictions["label"] != predictions["predicted_label"]]
    errors.to_csv(OUTPUT / "errors.csv", index=False, encoding="utf-8-sig")

    result = {
        "source": SOURCE,
        "text_sha256": data_hash,
        "seed": SEED,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "original_samples": original_count,
        "cleaned_samples": len(frame),
        "cleaned_spam_count": int(frame["target"].sum()),
        "split_sizes": {name: len(part) for name, part in splits.items()},
        "validation_results": validation_results,
        "selected_model": winner,
        "test_results": test_metrics,
    }
    (OUTPUT / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    joblib.dump(model, OUTPUT / "model.joblib")

    [[tn, fp], [fn, tp]] = test_metrics["confusion_matrix"]
    report = [
        "# 英文短信垃圾信息识别实验报告",
        "",
        f"数据来自 UCI SMS Spam Collection。原始数据共有 {original_count:,} 条，"
        f"清洗后有 {len(frame):,} 条，其中垃圾短信 {result['cleaned_spam_count']:,} 条。",
        f"训练、验证和测试分别使用 {len(training):,}、{len(validation):,} 和 {len(test):,} 条。"
        f"随机种子为 {SEED}。",
        "",
        "## 验证集模型对比",
        "",
        "| 模型 | Macro-F1 | 垃圾短信精确率 | 垃圾短信召回率 | 正常短信误报率 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in validation_results.items():
        report.append(
            f"| {name} | {metrics['macro_f1']:.4f} | {metrics['spam_precision']:.2%} | "
            f"{metrics['spam_recall']:.2%} | {metrics['ham_false_positive_rate']:.2%} |"
        )
    report.extend([
        "",
        "## 独立测试集",
        "",
        f"按验证集 Macro-F1 选中 **{winner}**。用训练与验证数据重新拟合后，"
        f"测试集 Macro-F1 为 **{test_metrics['macro_f1']:.4f}**。",
        f"垃圾短信精确率为 {test_metrics['spam_precision']:.2%}，"
        f"召回率为 {test_metrics['spam_recall']:.2%}，"
        f"正常短信误报率为 {test_metrics['ham_false_positive_rate']:.2%}，"
        f"Average Precision 为 {test_metrics['average_precision']:.4f}。",
        "",
        "| 真实类别 | 判为正常 | 判为垃圾 |",
        "| --- | ---: | ---: |",
        f"| 正常 | {tn} | {fp} |",
        f"| 垃圾 | {fn} | {tp} |",
        "",
        f"共有 {fp} 条正常短信被误报，{fn} 条垃圾短信被漏报，见 [错误记录](errors.csv)。",
        "本次结果使用单次固定划分。重复模板可能跨集合出现，且历史英文样本的评估结论适用于当前数据。",
        "",
        "运行版本与数据 SHA-256 见 [指标文件](metrics.json)。",
    ])
    (OUTPUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return result


def web_app():
    import streamlit as st

    st.set_page_config(page_title="短信垃圾信息识别")
    st.title("短信垃圾信息识别")
    st.caption("输入英文短信，查看模型判断和垃圾短信评分。")
    model_file = OUTPUT / "model.joblib"
    metrics_file = OUTPUT / "metrics.json"
    if not model_file.exists() or not metrics_file.exists():
        st.info("请先在项目目录运行 python app.py --train。")
        return

    with st.form("predict"):
        message = st.text_area("英文短信", NORMAL_SAMPLE, max_chars=5000)
        submitted = st.form_submit_button("识别短信", type="primary")

    if submitted:
        if not message.strip():
            st.warning("请输入短信内容。")
        else:
            model = joblib.load(model_file)
            predicted = model.predict([message])[0]
            score = model.predict_proba([message])[0, 1]
            st.subheader("疑似垃圾短信" if predicted else "正常短信")
            st.metric("垃圾短信评分", f"{score:.1%}")
            st.caption("基于历史英文样本训练，评分未经概率校准。")

    with st.expander("查看实验结果"):
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        comparison = pd.DataFrame.from_dict(metrics["validation_results"], orient="index")
        st.dataframe(comparison.drop(columns="confusion_matrix"))
        st.write(f"入选模型为 {metrics['selected_model']}。")
        test = metrics["test_results"]
        cols = st.columns(3)
        cols[0].metric("测试集 Macro-F1", f"{test['macro_f1']:.4f}")
        cols[1].metric("垃圾短信召回率", f"{test['spam_recall']:.2%}")
        cols[2].metric("正常短信误报率", f"{test['ham_false_positive_rate']:.2%}")


if __name__ == "__main__":
    if "--train" in sys.argv:
        print(json.dumps(train(), indent=2))
    else:
        web_app()
