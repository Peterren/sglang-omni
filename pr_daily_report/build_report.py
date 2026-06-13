# -*- coding: utf-8 -*-
"""Generate the SGLang-Omni daily PR study report (self-contained HTML)."""
import html
import re

# ---------------------------------------------------------------------------
# Tiny self-contained Python syntax highlighter (no JS / no CDN -> survives PDF)
# ---------------------------------------------------------------------------
KW = set(
    "def class return if elif else for while in not and or is import from as with "
    "try except finally raise lambda yield global nonlocal del pass break continue "
    "assert async await True False None".split()
)
BUILTINS = set(
    "self int float bool str list dict set tuple range len max min int64 float32 "
    "torch getattr setattr isinstance super print enumerate zip staticmethod".split()
)
TOKEN_RE = re.compile(
    r"(?P<comment>\#[^\n]*)"
    r"|(?P<string>\"[^\"\n]*\"|'[^'\n]*')"
    r"|(?P<number>\b\d+\.?\d*\b)"
    r"|(?P<name>\b[A-Za-z_][A-Za-z0-9_]*\b)"
    r"|(?P<other>.)",
    re.DOTALL,
)


def hl(code):
    code = code.strip("\n")
    out = []
    for m in TOKEN_RE.finditer(code):
        kind, text = m.lastgroup, m.group()
        esc = html.escape(text)
        if kind == "comment":
            out.append(f'<span class="c">{esc}</span>')
        elif kind == "string":
            out.append(f'<span class="s">{esc}</span>')
        elif kind == "number":
            out.append(f'<span class="m">{esc}</span>')
        elif kind == "name":
            if text in KW:
                out.append(f'<span class="k">{esc}</span>')
            elif text in BUILTINS:
                out.append(f'<span class="b">{esc}</span>')
            else:
                out.append(esc)
        else:
            out.append(esc)
    return "".join(out)


def code(snippet, lang="python"):
    return f'<pre class="code"><code>{hl(snippet)}</code></pre>'


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
CSS = """
:root{
  --bg:#0f1117; --card:#171a23; --ink:#e6e8ee; --muted:#9aa3b2; --line:#272b36;
  --accent:#7c9cff; --accent2:#52d1b2; --warn:#ffc16b; --pink:#ff8fb3;
  --codebg:#0b0d13;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei","Noto Sans CJK SC","WenQuanYi Zen Hei",Roboto,Helvetica,Arial,sans-serif;
  line-height:1.75; font-size:15.5px;
}
.wrap{max-width:980px; margin:0 auto; padding:34px 24px 80px}
header.top{
  border:1px solid var(--line); border-radius:18px; padding:30px 32px;
  background:linear-gradient(135deg,#1b2030,#141722); margin-bottom:26px;
}
header.top h1{margin:0 0 6px; font-size:27px; letter-spacing:.3px}
header.top .sub{color:var(--muted); font-size:14px}
.badges{margin-top:16px; display:flex; flex-wrap:wrap; gap:10px}
.badge{
  background:#10141f; border:1px solid var(--line); border-radius:999px;
  padding:5px 13px; font-size:12.5px; color:var(--muted);
}
.badge b{color:var(--accent)}
.stat-grid{display:flex; flex-wrap:wrap; gap:14px; margin:22px 0 8px}
.stat{
  flex:1 1 150px; background:var(--card); border:1px solid var(--line);
  border-radius:14px; padding:16px 18px;
}
.stat .num{font-size:26px; font-weight:700; color:var(--accent2)}
.stat .lab{font-size:12.5px; color:var(--muted); margin-top:2px}
h2.section{font-size:20px; margin:34px 0 14px; padding-bottom:8px; border-bottom:1px solid var(--line)}
.theme-table{width:100%; border-collapse:collapse; font-size:13.6px; margin:8px 0 6px}
.theme-table th,.theme-table td{border:1px solid var(--line); padding:9px 11px; text-align:left; vertical-align:top}
.theme-table th{background:#10141f; color:var(--muted); font-weight:600}
.theme-table td a{color:var(--accent); text-decoration:none}
.tag{display:inline-block; font-size:11.5px; padding:2px 9px; border-radius:999px; border:1px solid var(--line); color:var(--muted)}
.tag.arch{color:#9db4ff; border-color:#33406b}
.tag.perf{color:#52d1b2; border-color:#2c5b4f}
.tag.infra{color:#ffc16b; border-color:#5e4d2a}
.tag.tts{color:#ff8fb3; border-color:#5e2f42}

details.pr{
  background:var(--card); border:1px solid var(--line); border-radius:16px;
  margin:18px 0; overflow:hidden;
}
details.pr>summary{
  cursor:pointer; list-style:none; padding:18px 22px; display:block;
  background:#141824;
}
details.pr>summary::-webkit-details-marker{display:none}
.pr-id{font-size:12.5px; color:var(--accent); font-weight:700; letter-spacing:.4px}
.pr-title{font-size:17.5px; font-weight:650; margin:3px 0 6px}
.pr-meta{font-size:12.5px; color:var(--muted)}
.pr-meta a{color:var(--accent); text-decoration:none}
.pr-body{padding:6px 24px 26px}
.pr-body h3{
  font-size:15.5px; margin:24px 0 8px; color:#fff;
  display:flex; align-items:center; gap:9px;
}
.pr-body h3 .ic{
  font-size:12px; background:#10141f; border:1px solid var(--line);
  color:var(--accent2); border-radius:8px; width:26px; height:26px;
  display:inline-flex; align-items:center; justify-content:center; flex:none;
}
.pr-body h4{font-size:14px; margin:18px 0 6px; color:#cdd4e3}
.pr-body p{margin:8px 0}
.pr-body ul{margin:8px 0 8px; padding-left:22px}
.pr-body li{margin:5px 0}
.lead{color:var(--accent2); font-weight:600}
code.inl{
  background:#0b0d13; border:1px solid var(--line); border-radius:6px;
  padding:1px 6px; font-family:"SFMono-Regular",Consolas,"Liberation Mono",monospace;
  font-size:12.8px; color:#e9c08a;
}
pre.code{
  background:var(--codebg); border:1px solid var(--line); border-radius:12px;
  padding:14px 16px; overflow-x:auto; margin:12px 0;
  font-family:"SFMono-Regular",Consolas,"Liberation Mono",monospace;
  font-size:12.7px; line-height:1.6; color:#d7dce8; white-space:pre;
}
pre.code .k{color:#ff8fb3}
pre.code .b{color:#7c9cff}
pre.code .s{color:#9ad27d}
pre.code .c{color:#6b7384; font-style:italic}
pre.code .m{color:#e9c08a}
.note{
  border-left:3px solid var(--accent); background:#121622; border-radius:0 10px 10px 0;
  padding:11px 16px; margin:13px 0; font-size:14px;
}
.concept{
  border:1px solid var(--line); border-radius:12px; background:#10141f;
  padding:6px 18px 12px; margin:14px 0;
}
.concept>.h{
  font-size:13.5px; font-weight:700; color:var(--warn); margin:12px 0 4px;
  display:flex; gap:8px; align-items:center;
}
.concept .seg{margin:6px 0}
.concept .seg b{color:#cdd4e3}
.analogy{color:var(--accent2)}
.takeaway{
  border:1px dashed #33406b; border-radius:12px; background:#10141c;
  padding:8px 18px 14px; margin:14px 0;
}
.takeaway .h{color:#9db4ff; font-weight:700; font-size:13.5px; margin:10px 0 4px}
footer{margin-top:46px; padding-top:18px; border-top:1px solid var(--line); color:var(--muted); font-size:12.5px}
a{color:var(--accent)}
@media print{
  body{background:#fff; color:#1a1d26; font-size:11.5px; line-height:1.6}
  .wrap{max-width:100%; padding:0 6px}
  header.top{background:#f4f6fb; border-color:#d6dbe6}
  header.top h1,.pr-title,.pr-body h3,.pr-body h3 .ic{color:#11141c}
  .stat,.concept,.theme-table th,details.pr,details.pr>summary{background:#f7f8fc}
  details.pr,.stat,.concept,.theme-table td,.theme-table th,pre.code,.takeaway,.note{border-color:#d6dbe6}
  .pr-body h4{color:#2a2f3d}
  pre.code{background:#0b0d13; color:#d7dce8}            /* keep code dark for contrast */
  .badge,.pr-meta,.stat .lab,.muted,header.top .sub{color:#555}
  details.pr{break-inside:avoid; page-break-inside:avoid}
  .concept,pre.code,.takeaway{break-inside:avoid}
  h2.section{break-after:avoid}
  a{color:#1a4fd6}
}
"""

# ---------------------------------------------------------------------------
# helpers for building concept / takeaway blocks
# ---------------------------------------------------------------------------
def concept(title, what, why, here, analogy=None):
    a = f'<div class="seg analogy">🔗 <b>类比：</b>{analogy}</div>' if analogy else ""
    return f"""
    <div class="concept">
      <div class="h">📚 概念：{title}</div>
      <div class="seg"><b>它是什么 →</b> {what}</div>
      <div class="seg"><b>为什么需要它 →</b> {why}</div>
      <div class="seg"><b>在这个 PR 里怎么用 →</b> {here}</div>
      {a}
    </div>"""


def takeaway(items):
    lis = "".join(f"<li>{x}</li>" for x in items)
    return f'<div class="takeaway"><div class="h">💡 我能学到什么 / 延伸思考</div><ul>{lis}</ul></div>'


def h3(ic, txt):
    return f'<h3><span class="ic">{ic}</span>{txt}</h3>'


CARDS = []


def card(pr_id, title, meta, body):
    CARDS.append(
        f"""
    <details class="pr" open>
      <summary>
        <div class="pr-id">{pr_id}</div>
        <div class="pr-title">{title}</div>
        <div class="pr-meta">{meta}</div>
      </summary>
      <div class="pr-body">{body}</div>
    </details>"""
    )
