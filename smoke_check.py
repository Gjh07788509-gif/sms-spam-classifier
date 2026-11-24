"""Run after training to validate a saved model and both demo outcomes."""
import joblib
from streamlit.testing.v1 import AppTest

from app import NORMAL_SAMPLE, OUTPUT, ROOT, SPAM_SAMPLE


def main():
    model = joblib.load(OUTPUT / "model.joblib")
    assert model.predict([NORMAL_SAMPLE, SPAM_SAMPLE]).tolist() == [0, 1]
    page = AppTest.from_file(str(ROOT / "app.py")).run()
    for message, expected in [(NORMAL_SAMPLE, "正常短信"), (SPAM_SAMPLE, "疑似垃圾短信")]:
        page.text_area[0].set_value(message)
        page.button[0].click().run()
        assert not page.exception
        assert page.subheader[0].value == expected
    print("Model and Streamlit demo passed both example predictions.")


if __name__ == "__main__":
    main()
