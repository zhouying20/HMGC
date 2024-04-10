import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix


def calc_classification_metrics(preds, labels, target_names):
    preds = np.array(preds)
    labels = np.array(labels)

    auc = roc_auc_score(labels, preds)
    clf_report = classification_report(
        labels, preds, target_names=target_names, output_dict=True
    )
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel() / len(labels)

    # return {
    #     "auc_roc": auc,
    #     "acc": clf_report['accuracy'],
    #     "avg_prec": clf_report['weighted avg']['precision'],
    #     "avg_recall": clf_report['weighted avg']['recall'],
    #     "avg_f1": clf_report['weighted avg']['f1-score'],
    #     "gpt_prec": clf_report['gpt']['precision'],
    #     "gpt_recall": clf_report['gpt']['recall'],
    #     "gpt_f1": clf_report['gpt']['f1-score'],
    #     "human_prec": clf_report['human']['precision'],
    #     "human_recall": clf_report['human']['recall'],
    #     "human_f1": clf_report['human']['f1-score'],
    #     "true_positive": tp,
    #     "false_positive": fp,
    #     "true_negative": tn,
    #     "false_negative": fn,
    # }
    return {
        "auc_roc": auc,
        "acc": clf_report["accuracy"],
        "gpt_acc": clf_report["gpt"]["recall"],
        "human_acc": clf_report["human"]["recall"],
        "avg_f1": clf_report["weighted avg"]["f1-score"],
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
    }