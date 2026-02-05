import os, re
from flask import Flask, render_template, request, redirect, url_for
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

#  Config 
MODEL_DIR = os.environ.get("MODEL_DIR", "artifacts/t5_base_merged_wikiasp_edu")
MAX_SRC_LEN = 512              # T5-base max input tokens
CHUNK_OVERLAP = 64             # token overlap between chunks
MAX_TGT_LEN = 256              # allow longer summaries
MIN_NEW_TOKENS = 60            # force non-trivial length
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Loading the model 
print(f"Loading model from {MODEL_DIR} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()
print("Model loaded successfully.")

app = Flask(__name__)

#  Utilities 
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

def split_into_sentences(text: str):

    sents = _SENT_SPLIT.split(text.strip())

    merged = []
    buf = []
    for s in sents:
        buf.append(s)
        if len(" ".join(buf)) > 40: 
            merged.append(" ".join(buf))
            buf = []
    if buf:
        merged.append(" ".join(buf))
    return merged

def pack_sentences_to_chunks(sents, max_tokens, overlap):
    chunks, cur, cur_tokens = [], [], 0
    for s in sents:
        t = tokenizer.encode(s, add_special_tokens=False)
        if cur_tokens + len(t) > max_tokens:
            if cur:
                chunks.append(" ".join(cur))

                tail = []
                tail_tokens = 0
                for sent in reversed(cur):
                    tt = tokenizer.encode(sent, add_special_tokens=False)
                    if tail_tokens + len(tt) > overlap:
                        break
                    tail.append(sent)
                    tail_tokens += len(tt)
                cur = list(reversed(tail))
                cur_tokens = sum(len(tokenizer.encode(x, add_special_tokens=False)) for x in cur)
        cur.append(s)
        cur_tokens += len(t)
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def _generate_summary(text: str, max_new_tokens=MAX_TGT_LEN, min_new_tokens=MIN_NEW_TOKENS):
    enc = tokenizer(
        "summarize: " + text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SRC_LEN,
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(
            **enc,
            num_beams=6,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            repetition_penalty=1.05,
            early_stopping=True,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def summarize_long_text(text: str) -> str:

    if len(tokenizer.encode(text, add_special_tokens=False)) <= MAX_SRC_LEN:
        return _generate_summary(text)


    sents = split_into_sentences(text)
    chunks = pack_sentences_to_chunks(sents, MAX_SRC_LEN, CHUNK_OVERLAP)

    partials = []
    for c in chunks:
        partials.append(_generate_summary(c, max_new_tokens=200, min_new_tokens=50))

    stitched = " ".join(partials)

    final = _generate_summary(stitched, max_new_tokens=MAX_TGT_LEN, min_new_tokens=MIN_NEW_TOKENS)
    return final

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    summary, text = None, ""
    if request.method == "POST":
        if "clear" in request.form:
            return redirect(url_for("index"))
        text = request.form.get("text", "").strip()
        if text:
            summary = summarize_long_text(text)
    return render_template("index.html", summary=summary, text=text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
