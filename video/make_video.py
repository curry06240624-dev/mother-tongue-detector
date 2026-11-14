"""Makes the 2-minute demo video: AI narration (edge-tts), a real screen recording of app.py, cards, captions.

Run from the project folder:  python video/make_video.py
Output: video/mother_tongue_detector_demo.mp4
"""
import asyncio, json, os, subprocess, sys, time

sys.path.insert(0, os.getcwd())
WORK = os.path.join("video", "work")
os.makedirs(WORK, exist_ok=True)
W, H = 1280, 720
VOICE = "en-US-AndrewNeural"

# id, text spoken (say), text shown as caption (show)
LINES = [
    ("A", "In Shin-joo, Taiwan, people speak Mandarin, Hakka and Taiwanese Hokkien. All three are written with the same "
          "Chinese characters, and software can't tell them apart. A popular language detector calls about half of these "
          "sentences Korean. But every tool that reads, translates or routes text needs to know the language first.",
          "In Hsinchu, Taiwan, people speak Mandarin, Hakka and Taiwanese Hokkien. All three are written with the same "
          "Chinese characters, and software can't tell them apart. A popular language detector calls about half of these "
          "sentences Korean. But every tool that reads, translates or routes text needs to know the language first."),
    ("B1", "Here's a real Taiwanese Hokkien sentence from our test set. The model never saw it. It answers: Taiwanese "
           "Hokkien, ninety-seven percent sure. langdetect says Korean.", None),
    ("B2", "Here's the same sentence in Mandarin. Same meaning, different language, and the model tells them apart.", None),
    ("B3", "Now Hakka. The colours show which characters decided it: mostly small grammar words, the ones every "
           "sentence needs.", None),
    ("B4", "Cantonese works too.", None),
    ("B5", "And when a text is too short to tell, it says so instead of guessing.", None),
    ("C1", "We trained on example sentences from the Ministry of Education's Hokkien and Hakka dictionaries. Each one "
           "comes with its Mandarin translation, so the model learns the language, not the topic.", None),
    ("C2", "The model is simple: character pairs and a logistic regression. On held-out test text it is right "
           "eighty-six percent of the time, and ninety-four percent with four characters or more. "
           "A simple rule gets fifty-five, and langdetect gets twenty-one.",
           "The model is simple: character pairs and a logistic regression. On held-out test text it is right "
           "86% of the time, and 94% with four characters or more. A simple rule gets 55%, and langdetect gets 21%."),
    ("D1", "Where it fails: Hakka written in everyday spelling. This sentence uses a word for 'not' that Cantonese also "
           "uses, so the model says Cantonese. That's wrong.", None),
    ("D2", "Next, we want to test it with Hakka and Hokkien speakers in Shin-joo, and learn how people really write.",
           "Next, we want to test it with Hakka and Hokkien speakers in Hsinchu, and learn how people really write."),
]
SHOW = {k: (show or say) for k, say, show in LINES}

# Demo steps typed into the app (all real sentences; the last one is a real Tatoeba Hakka sentence)
STEPS = [("B1", "阮先生娘生做真媠。"), ("B2", "我們的師母長得很美。"), ("B3", "冷風對窗門个縫仔項鑽入來。"),
         ("B4", "我喺屋企同屋企人一齊嗰陣最開心。"), ("B5", "食飽未？"), ("D1", "阿薩米唔聽涯。")]


def run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
                         check=True, capture_output=True, text=True).stdout
    return float(json.loads(out)["format"]["duration"])


# ---------- 1. narration ----------
async def make_audio():
    import edge_tts
    for k, say, _ in LINES:
        path = os.path.join(WORK, f"{k}.mp3")
        if not os.path.exists(path):
            await edge_tts.Communicate(say, VOICE, rate="+4%").save(path)

asyncio.run(make_audio())
DUR = {k: duration(os.path.join(WORK, f"{k}.mp3")) for k, _, _ in LINES}
print("narration seconds:", {k: round(v, 1) for k, v in DUR.items()}, "total", round(sum(DUR.values()), 1))

# ---------- 2. cards (HTML -> PNG) ----------
CSS = """
body{margin:0;width:1280px;height:720px;background:#0f1c2e;color:#f3f6fa;font-family:'Segoe UI','Microsoft JhengHei',sans-serif;
display:flex;flex-direction:column;justify-content:center;padding:0 90px;box-sizing:border-box}
h1{font-size:54px;margin:0 0 10px} h2{font-size:38px;margin:0 0 28px;color:#9cc7ff} p{font-size:26px;line-height:1.45;margin:8px 0}
.zh{font-family:'Microsoft JhengHei',sans-serif} .row{display:flex;align-items:center;font-size:30px;margin:12px 0}
.row .s{flex:0 0 520px} .row .l{flex:0 0 260px;color:#cfe3ff} .tag{padding:6px 16px;border-radius:8px;font-size:24px}
.bad{background:#7a1f2b;color:#ffd9de} .ok{background:#24543a;color:#d6f5e2} .small{font-size:20px;color:#9fb0c4;margin-top:34px}
.bar{display:flex;align-items:center;margin:14px 0;font-size:26px} .bar .n{flex:0 0 300px} .bar .b{height:44px;border-radius:6px;margin-right:14px}
"""

def card_html(body):
    return f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"

CARDS = {
    "title": card_html("""<h1>Mother-Tongue Detector <span class='zh'>· 母語偵測器</span></h1>
        <h2>Telling Taiwanese Hokkien, Hakka, Mandarin and Cantonese apart</h2>
        <p class='small'>Veritas AI International High School Hackathon 2026 · Languages track</p>"""),
    "problem": card_html("""<h2>Same characters. Different languages.</h2>
        <div class='row'><span class='s zh'>阮先生娘生做真媠。</span><span class='l'>Taiwanese Hokkien</span><span class='tag bad'>langdetect: Korean</span></div>
        <div class='row'><span class='s zh'>冷風對窗門个縫仔項鑽入來。</span><span class='l'>Hakka</span><span class='tag bad'>langdetect: Korean</span></div>
        <div class='row'><span class='s zh'>我們的師母長得很美。</span><span class='l'>Mandarin</span><span class='tag ok'>langdetect: Chinese</span></div>
        <p class='small'>langdetect labels 44–49% of our Hokkien and Hakka test sentences "Korean" (and 49% of the Mandarin ones).</p>"""),
    "data": card_html("""<h2>Data: Taiwan's Ministry of Education dictionaries</h2>
        <p>14,960 Hokkien + 18,705 Hakka example sentences, each with its Mandarin translation</p>
        <div class='row'><span class='s zh' style='flex-basis:560px'>冷風對窗門个縫仔項鑽入來。</span><span class='l'>Hakka</span></div>
        <div class='row'><span class='s zh' style='flex-basis:560px'>寒風從窗戶的縫隙中鑽了進來。</span><span class='l'>Mandarin</span></div>
        <p>+ 17,120 Cantonese sentences from Tatoeba</p>
        <p class='small'>Split by dictionary entry (70% train / 15% validation / 15% test), so a sentence and its translation never cross over.</p>"""),
    "results": card_html("""<h2>Accuracy on 9,141 held-out test texts</h2>
        <div class='bar'><span class='n'>Always "Mandarin"</span><span class='b' style='width:%dpx;background:#5b6b80'></span>42%%</div>
        <div class='bar'><span class='n'>langdetect</span><span class='b' style='width:%dpx;background:#5b6b80'></span>21%%</div>
        <div class='bar'><span class='n'>Simple character rule</span><span class='b' style='width:%dpx;background:#5b6b80'></span>55%%</div>
        <div class='bar'><span class='n'><b>Our model</b></span><span class='b' style='width:%dpx;background:#3d8fe0'></span><b>86%%</b></div>
        <p class='small'>Our model: character pairs (TF-IDF) + logistic regression. 94%% on texts with 4+ characters; 54%% on 2–3 characters.</p>"""
        % (int(7.2 * 42), int(7.2 * 21), int(7.2 * 55), int(7.2 * 86))),
    "end": card_html("""<h2>Next: test it with Hakka and Hokkien speakers in Hsinchu</h2>
        <p>Main weakness: Hakka in everyday spelling (<span class='zh'>涯</span> for <span class='zh'>𠊎</span> "I",
        <span class='zh'>唔</span> for <span class='zh'>毋</span> "not") is often called Cantonese.</p>
        <p class='small'>Mother-Tongue Detector · Veritas AI International High School Hackathon 2026<br>
        AI tools used: Claude Code (code, write-up, script) · Microsoft Edge text-to-speech (this voice)</p>"""),
}

# ---------- 3. screen recording of the real app ----------
import gradio as gr
from playwright.sync_api import sync_playwright
import app

app.demo.launch(theme=gr.themes.Soft(), prevent_thread_lock=True, server_port=7862, quiet=True)
marks = {}
with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome")
    page = browser.new_page(viewport={"width": W, "height": H})
    for name, html in CARDS.items():
        page.set_content(html)
        page.wait_for_timeout(300)
        page.screenshot(path=os.path.join(WORK, f"card_{name}.png"))
    page.close()
    ctx = browser.new_context(viewport={"width": W, "height": H}, record_video_dir=WORK,
                              record_video_size={"width": W, "height": H})
    page = ctx.new_page()
    t0 = time.monotonic()
    page.goto("http://127.0.0.1:7862/")
    page.wait_for_selector("textarea")
    page.wait_for_timeout(1500)
    box = page.locator("textarea").first
    for key, sentence in STEPS:
        box.fill("")
        start = time.monotonic() - t0
        box.press_sequentially(sentence, delay=110)
        page.get_by_role("button", name="Detect language").click()
        page.wait_for_timeout(600)
        left = DUR[key] + 1.0 - (time.monotonic() - t0 - start)
        page.wait_for_timeout(max(left, 1.2) * 1000)
        marks[key] = (start, time.monotonic() - t0)
    video_path = page.video.path()
    ctx.close()
    browser.close()
app.demo.close()
REC = os.path.join(WORK, "recording.mp4")
run(["ffmpeg", "-y", "-i", video_path, "-r", "30", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", REC])
print("recording marks:", {k: (round(a, 1), round(b, 1)) for k, (a, b) in marks.items()})

# ---------- 4. timeline ----------
segments = []   # (kind, source, start_in_source, length)
cues = []       # (line id, start in final video)
t = 0.0
def add_card(name, length):
    global t
    segments.append(("card", os.path.join(WORK, f"card_{name}.png"), 0, length)); t += length
def add_rec(a, b):
    global t
    segments.append(("rec", REC, a, b - a)); t += b - a

add_card("title", 3.0)
cues.append(("A", t + 0.3)); add_card("problem", DUR["A"] + 0.9)
b_start = max(marks["B1"][0] - 0.3, 0)        # start the cut a moment early, in case the recording clock lags
for key in ["B1", "B2", "B3", "B4", "B5"]:
    cues.append((key, t + (marks[key][0] - b_start) + 0.2))
add_rec(b_start, marks["B5"][1])
cues.append(("C1", t + 0.2)); add_card("data", DUR["C1"] + 0.6)
cues.append(("C2", t + 0.2)); add_card("results", DUR["C2"] + 0.8)
d_start = marks["D1"][0] - 0.3
cues.append(("D1", t + 0.5)); add_rec(d_start, marks["D1"][1])
cues.append(("D2", t + 0.2)); add_card("end", DUR["D2"] + 2.0)
TOTAL = t
print(f"total length: {TOTAL:.1f} s")

# ---------- 5. render pieces, then join ----------
parts = []
for i, (kind, src, a, length) in enumerate(segments):
    out = os.path.join(WORK, f"part_{i}.mp4")
    if kind == "card":
        run(["ffmpeg", "-y", "-loop", "1", "-t", f"{length:.3f}", "-i", src, "-r", "30", "-vf", f"scale={W}:{H},format=yuv420p",
             "-c:v", "libx264", "-crf", "16", out])
    else:
        run(["ffmpeg", "-y", "-ss", f"{a:.3f}", "-t", f"{length:.3f}", "-i", src, "-r", "30",
             "-vf", f"scale={W}:{H},format=yuv420p", "-c:v", "libx264", "-crf", "16", out])
    parts.append(out)
with open(os.path.join(WORK, "parts.txt"), "w") as f:
    f.writelines(f"file '{os.path.basename(p)}'\n" for p in parts)
run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", os.path.join(WORK, "parts.txt"), "-c", "copy",
     os.path.join(WORK, "silent.mp4")])

# narration track: each line placed at its cue time
inputs, filters = [], []
for n, (key, start) in enumerate(cues):
    inputs += ["-i", os.path.join(WORK, f"{key}.mp3")]
    ms = int(start * 1000)
    filters.append(f"[{n}:a]adelay={ms}|{ms}[a{n}]")
mix = "".join(f"[a{n}]" for n in range(len(cues)))
filters.append(f"{mix}amix=inputs={len(cues)}:normalize=0,apad,atrim=0:{TOTAL:.3f}[out]")
run(["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "[out]", "-c:a", "aac", "-b:a", "160k",
     os.path.join(WORK, "narration.m4a")])

# captions: split each line into chunks of about 12 words, spread over the line's duration
def fmt(s):
    h, rem = divmod(s, 3600); m, sec = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{sec:05.2f}"
events = []
for key, start in cues:
    words = SHOW[key].split()
    chunks = [" ".join(words[i:i + 12]) for i in range(0, len(words), 12)]
    total_chars = sum(len(c) for c in chunks)
    s = start
    for c in chunks:
        d = DUR[key] * len(c) / total_chars
        events.append(f"Dialogue: 0,{fmt(s)},{fmt(s + d)},Default,,0,0,0,,{c}")
        s += d
ass = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Segoe UI,30,&H00FFFFFF,&H00FFFFFF,&H00000000,&H99000000,1,0,0,0,100,100,0,0,3,1,0,2,60,60,26,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""" + "\n".join(events) + "\n"
with open(os.path.join(WORK, "captions.ass"), "w", encoding="utf-8") as f:
    f.write(ass)

final = os.path.abspath(os.path.join("video", "mother_tongue_detector_demo.mp4"))
subprocess.run(["ffmpeg", "-y", "-i", "silent.mp4", "-i", "narration.m4a", "-vf", "subtitles=captions.ass",
                "-c:v", "libx264", "-crf", "18", "-preset", "slow", "-c:a", "copy", "-shortest", "-movflags", "+faststart",
                final], check=True, capture_output=True, cwd=WORK)
print(f"wrote {final}  ({duration(final):.1f} s, {os.path.getsize(final) / 1e6:.1f} MB)")
