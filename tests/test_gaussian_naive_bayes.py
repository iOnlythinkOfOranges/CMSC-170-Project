# During the preparation of this work, the author used Claude Sonnet 5.5 here in order to build this test case and fix errors upon running the code. 
# After using this tool, the author reviewed and edited the content as needed and take full responsibility for the content.

"""Checks the model against the worked example, which was calculated by hand."""
import os
import sys

# Let this file import from the src folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.gaussian_naive_bayes import GaussianNaiveBayes

def test_gaussian_naive_bayes():
    # 10 made-up days: [humidity, sunlight], and whether it rained the next day
    X = [[90, 10], [86, 12], [88, 8], [92, 10],
         [70, 16], [74, 20], [66, 18], [72, 20], [68, 16], [70, 18]]
    y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

    model = GaussianNaiveBayes().fit(X, y)

    # Priors: 6 dry days and 4 rainy days out of 10
    assert abs(model.priors_[0] - 0.6) < 1e-9
    assert abs(model.priors_[1] - 0.4) < 1e-9

    # Means and variances computed by hand
    assert abs(model.means_[1][0] - 89) < 1e-6      # rain: humidity mean
    assert abs(model.means_[1][1] - 10) < 1e-6      # rain: sunlight mean
    assert abs(model.means_[0][0] - 70) < 1e-6      # dry: humidity mean
    assert abs(model.means_[0][1] - 18) < 1e-6      # dry: sunlight mean
    assert abs(model.vars_[1][0] - 5) < 1e-3        # rain: humidity variance
    assert abs(model.vars_[0][0] - 6.6667) < 1e-3   # dry: humidity variance

    # The in-between day (humidity 80, sunlight 14): P(rain) is about 0.152 by hand
    p_mid = model.predict_proba([[80, 14]])[0]
    assert abs(p_mid - 0.152) < 0.005, p_mid

    # Default threshold 0.5 -> dry. A low threshold of 0.1 flips it to rain
    assert model.predict([[80, 14]])[0] == 0
    assert model.predict([[80, 14]], threshold=0.1)[0] == 1

    # Clear cases
    assert model.predict([[91, 9]])[0] == 1         # very humid, little sun -> rain
    assert model.predict([[68, 19]])[0] == 0        # drier, sunny -> dry

    # Output format: a list of probabilities between 0 and 1
    probs = model.predict_proba([[91, 9], [68, 19], [80, 14]])
    assert isinstance(probs, list) and all(0.0 <= p <= 1.0 for p in probs)

    # fit() must complain if a class is missing.
    try:
        GaussianNaiveBayes().fit([[1.0], [2.0]], [0, 0])
        raise SystemExit("Expected an error when a class is missing")
    except ValueError:
        pass


if __name__ == "__main__":
    test_gaussian_naive_bayes()
    print("All Gaussian Naive Bayes tests passed.")