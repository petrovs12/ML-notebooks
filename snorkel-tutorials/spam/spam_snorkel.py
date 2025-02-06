import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


app._unparsable_cell(
    r"""
    #%matplotlib inline 
    import os



    # Make sure we're running from the spam/ directory
    if os.path.basename(os.getcwd()) == \"snorkel-tutorials\":
        os.chdir(\"spam\")

    # Turn off TensorFlow logging messages
    os.environ[\"TF_CPP_MIN_LOG_LEVEL\"] = \"3\"

    # For reproducibility
    os.environ[\"PYTHONHASHSEED\"] = \"0\"
    ! python -m spacy download en_core_web_sm
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    ! python -m spacy download en_core_web_sm
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    !python -m spacy download en_core_web_sm
    """,
    name="_"
)


@app.cell
def _():
    from utils import load_spam_dataset

    df_train, df_test = load_spam_dataset()

    # We pull out the label vectors for ease of use later
    Y_test = df_test.label.values
    return Y_test, df_test, df_train, load_spam_dataset


@app.cell
def _(Y_test):
    Y_test
    return


@app.cell
def _(df_train):
    Y_train = df_train.label.values
    return (Y_train,)


@app.cell
def _(df_train):
    X_train = df_train.text.values
    return (X_train,)


@app.cell
def _(df_train):
    ABSTAIN = -1
    HAM = 0
    SPAM = 1
    df_train[["author", "text", "video"]].sample(20, random_state=2)
    return ABSTAIN, HAM, SPAM


@app.cell
def _(ABSTAIN, HAM, SPAM, df_train):
    from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis, LabelingFunction
    from textblob import TextBlob
    import spacy

    from snorkel.preprocess import preprocessor
    @labeling_function()
    def check(x):
        return SPAM if "check" in x.text.lower() else ABSTAIN
    @labeling_function()
    def check_out(x):
        return SPAM if "check out" in x.text.lower() else ABSTAIN

    import re


    @labeling_function()
    def regex_check_out(x):
        return SPAM if re.search(r"check.*out", x.text, flags=re.I) else ABSTAIN

    def keyword_lookup(x, keywords, label):
        if any(word in x.text.lower() for word in keywords):
            return label
        return ABSTAIN


    def make_keyword_lf(keywords, label=SPAM):
        return LabelingFunction(
            name=f"keyword_{keywords[0]}",
            f=keyword_lookup,
            resources=dict(keywords=keywords, label=label),
        )


    """Spam comments talk about 'my channel', 'my video', etc."""
    keyword_my = make_keyword_lf(keywords=["my"])

    """Spam comments ask users to subscribe to their channels."""
    keyword_subscribe = make_keyword_lf(keywords=["subscribe"])

    """Spam comments post links to other channels."""
    keyword_link = make_keyword_lf(keywords=["http"])

    """Spam comments make requests rather than commenting."""
    keyword_please = make_keyword_lf(keywords=["please", "plz"])

    """Ham comments actually talk about the video's content."""
    keyword_song = make_keyword_lf(keywords=["song"], label=HAM)

    @preprocessor(memoize=True)
    def textblob_sentiment(x):
        scores = TextBlob(x.text)
        x.polarity = scores.sentiment.polarity
        x.subjectivity = scores.sentiment.subjectivity
        return x

    @labeling_function(pre=[textblob_sentiment])
    def textblob_polarity(x):
        return HAM if x.polarity > 0.9 else ABSTAIN

    @labeling_function(pre=[textblob_sentiment])
    def textblob_subjectivity(x):
        return HAM if x.subjectivity >= 0.5 else ABSTAIN



    @labeling_function()
    def short_comment(x):
        """Ham comments are often short, such as 'cool video!'"""
        return HAM if len(x.text.split()) < 5 else ABSTAIN

    from snorkel.preprocess.nlp import SpacyPreprocessor

    # The SpacyPreprocessor parses the text in text_field and
    # stores the new enriched representation in doc_field
    spacy = SpacyPreprocessor(text_field="text", doc_field="doc", memoize=True)

    @labeling_function(pre=[spacy])
    def has_person(x):
        """Ham comments mention specific people and are short."""
        if len(x.doc) < 20 and any([ent.label_ == "PERSON" for ent in x.doc.ents]):
            return HAM
        else:
            return ABSTAIN

    from snorkel.labeling.lf.nlp import nlp_labeling_function


    @nlp_labeling_function()
    def has_person_nlp(x):
        """Ham comments mention specific people and are short."""
        if len(x.doc) < 20 and any([ent.label_ == "PERSON" for ent in x.doc.ents]):
            return HAM
        else:
            return ABSTAIN
    lfs=[check, check_out, regex_check_out, textblob_polarity, textblob_subjectivity, keyword_my, keyword_subscribe, keyword_link, keyword_please, keyword_song, short_comment, has_person, has_person_nlp]
    checking_applier = PandasLFApplier(lfs=lfs)
    L_train = checking_applier.apply(df=df_train)
    L_train.shape
    return (
        LFAnalysis,
        L_train,
        LabelingFunction,
        PandasLFApplier,
        SpacyPreprocessor,
        TextBlob,
        check,
        check_out,
        checking_applier,
        has_person,
        has_person_nlp,
        keyword_link,
        keyword_lookup,
        keyword_my,
        keyword_please,
        keyword_song,
        keyword_subscribe,
        labeling_function,
        lfs,
        make_keyword_lf,
        nlp_labeling_function,
        preprocessor,
        re,
        regex_check_out,
        short_comment,
        spacy,
        textblob_polarity,
        textblob_sentiment,
        textblob_subjectivity,
    )


@app.cell
def _(L_train):
    L_train
    return


@app.cell
def _():
         #   coverage_check_out, coverage_check = (L_train != ABSTAIN).mean(axis=0)
          #  print(f"check_out coverage: {coverage_check_out * 100:.1f}%")
           # print(f"check coverage: {coverage_check * 100:.1f}%")
    return


@app.cell
def _(LFAnalysis, L_train, lfs):
    LFAnalysis(L=L_train, lfs=lfs).lf_summary()
    return


@app.cell
def _(ABSTAIN, L_train):
    import matplotlib.pyplot as plt




    def plot_label_frequency(L):
        plt.hist((L != ABSTAIN).sum(axis=1), density=True, bins=range(L.shape[1]))
        plt.xlabel("Number of labels")
        plt.ylabel("Fraction of dataset")
        plt.show()


    plot_label_frequency(L_train)
    return plot_label_frequency, plt


@app.cell
def _(L_train):
    from snorkel.labeling.model import MajorityLabelVoter

    majority_model = MajorityLabelVoter()
    preds_train = majority_model.predict(L=L_train)


    return MajorityLabelVoter, majority_model, preds_train


@app.cell
def _(L_train):
    from snorkel.labeling.model import LabelModel

    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)
    return LabelModel, label_model


@app.cell
def _(L_train, SPAM, label_model, majority_model, plt):


    def plot_probabilities_histogram(Y):
        plt.hist(Y, bins=10)
        plt.xlabel("Probability of SPAM")
        plt.ylabel("Number of data points")
        plt.show()


    probs_train = label_model.predict_proba(L=L_train)
    probs_train_majority = majority_model.predict_proba(L = L_train)
    plot_probabilities_histogram(probs_train[:, SPAM])
    return plot_probabilities_histogram, probs_train, probs_train_majority


@app.cell
def _(probs_train):
    probs_train 
    return


@app.cell
def _(probs_train_majority):
    probs_train_majority
    return


@app.cell
def _(probs_train):
    import numpy as np
    np.sum(probs_train,axis= 1)
    return (np,)


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
