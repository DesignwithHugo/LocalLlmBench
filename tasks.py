"""Quality task set. Auto-graded where possible; 'judge' items are scored by local judge models.
Every grader returns (score 0..1, note)."""
import json, os, re, subprocess, sys, tempfile, unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "assets")


def _read(p):
    return open(os.path.join(A, p), encoding="utf-8").read()


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def norm(s):
    return strip_accents(s).lower()


ES_WORDS = set("el la los las de que y en un una por con para no se es del al lo como más pero su sus le ya".split())
EN_WORDS = set("the and of to in is that for it with as on are be this by not or from".split())


def lang_ratio(text):
    w = re.findall(r"[a-záéíóúñü]+", text.lower())
    if not w:
        return 0, 0
    return sum(x in ES_WORDS for x in w) / len(w), sum(x in EN_WORDS for x in w) / len(w)


def is_spanish(t):
    es, en = lang_ratio(t)
    return es > en and es > 0.08


def is_english(t):
    es, en = lang_ratio(t)
    return en > es and en > 0.08


def words(t):
    return re.findall(r"\b[\wáéíóúñüÁÉÍÓÚÑÜ'-]+\b", t)


def last_tag(text, tag):
    m = re.findall(tag + r"\s*:\s*(.+)", text, flags=re.I)
    return m[-1].strip() if m else None


def to_num(s):
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "")
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group()) if m else None


def extract_json(text):
    t = text.strip()
    fenced = False
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t, fenced = m.group(1).strip(), True
    for cand in (t, t[t.find("{"):t.rfind("}") + 1] if "{" in t else "", t[t.find("["):t.rfind("]") + 1] if "[" in t else ""):
        try:
            return json.loads(cand), fenced
        except Exception:
            continue
    return None, fenced


def extract_code(text):
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return max(m, key=len) if m else text


def run_py_tests(code, tests, timeout=20):
    src = code + "\n\n" + "_ok=0\n" + "".join(
        f"try:\n    assert {t}\n    _ok+=1\nexcept Exception as _e:\n    pass\n" for t in tests) + "print('PASSED', _ok)\n"
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.py")
        open(p, "w").write(src)
        try:
            r = subprocess.run([sys.executable, "-I", p], capture_output=True, text=True, timeout=timeout, cwd=d)
            m = re.search(r"PASSED (\d+)", r.stdout)
            n = int(m.group(1)) if m else 0
        except subprocess.TimeoutExpired:
            n = 0
    return n / len(tests), f"{n}/{len(tests)} tests"


# ---------------------------------------------------------------- graders

def g_bullets4(ans):
    lines = [l.strip() for l in ans.strip().splitlines() if l.strip()]
    bul = [l for l in lines if re.match(r"^[-*•]\s", l)]
    other = [l for l in lines if l not in bul]
    s = 0.0
    s += 0.4 if len(bul) == 4 else 0
    s += 0.2 if not other else 0
    s += 0.2 if bul and all(len(words(b)) < 16 for b in bul) else 0
    s += 0.2 if is_spanish(ans) else 0
    return s, f"{len(bul)} bullets, {len(other)} extra lines"


def g_json_only(ans):
    obj, fenced = extract_json(ans)
    if not isinstance(obj, dict):
        return 0, "not JSON"
    s = 0.5 if fenced else 0.7
    ok_hex = isinstance(obj.get("color"), str) and re.fullmatch(r"#?[0-9a-fA-F]{6}", obj["color"].strip())
    ok_size = isinstance(obj.get("size"), int) and 12 <= obj["size"] <= 24
    s += 0.1 * bool(ok_hex) + 0.1 * isinstance(obj.get("font"), str) + 0.1 * bool(ok_size)
    return min(s, 1.0), ("fenced" if fenced else "raw") + f", hex={bool(ok_hex)} size={bool(ok_size)}"


def g_lower_fin(ans):
    a = ans.strip()
    s = 0.0
    s += 0.4 if a == a.lower() else 0
    s += 0.4 if norm(a).rstrip().endswith("fin.") else 0
    s += 0.2 if is_spanish(a) else 0
    return s, ""


def g_25words(ans):
    n = len(words(ans))
    return (1.0 if n == 25 else 0.5 if abs(n - 25) <= 2 else 0.0), f"{n} words"


def g_forbidden(ans):
    a = norm(ans)
    bad = [w for w in ("espacio", "letras") if re.search(r"\b" + w + r"\b", a)]
    s = (0.0 if bad else 0.6) + (0.2 if len(words(ans)) >= 20 else 0) + (0.2 if is_spanish(ans) else 0)
    return s, ("used: " + ",".join(bad)) if bad else "ok"


def g_factura(ans):
    obj, _ = extract_json(ans)
    if not isinstance(obj, dict):
        return 0, "not JSON"
    exp = {"folio": "F-2291", "fecha": "2026-03-03", "subtotal": 6040.0, "iva": 966.4, "total": 7006.4}
    hit = 0
    for k, v in exp.items():
        got = obj.get(k)
        if isinstance(v, float):
            g = to_num(str(got)) if got is not None else None
            hit += g is not None and abs(g - v) < 0.02
        else:
            hit += isinstance(got, str) and got.strip().upper() == v.upper()
    hit += isinstance(obj.get("emisor"), str) and "papeleria monterrey" in norm(obj["emisor"])
    hit += isinstance(obj.get("receptor"), str) and "rivas" in norm(obj["receptor"])
    return hit / 7, f"{hit}/7 fields"


def g_meetings(ans):
    obj, _ = extract_json(ans)
    if isinstance(obj, dict):
        obj = next((v for v in obj.values() if isinstance(v, list)), None)
    if not isinstance(obj, list):
        return 0, "no list"
    # day + time must be exact; the topic only has to be one word taken from that meeting's sentence
    exp = [("tuesday", "09:30", ("budget", "review")), ("wednesday", "14:00", ("portfolio", "critique")),
           ("friday", "17:15", ("launch", "rehearsal"))]
    hit = 0.0
    for day, tm, topics in exp:
        best = 0.0
        for o in obj:
            if not isinstance(o, dict):
                continue
            vals = {k.lower(): norm(str(v)) for k, v in o.items()}
            if day in vals.get("day", "") and tm.lstrip("0") in vals.get("time", "").lstrip("0"):
                best = max(best, 1.0 if any(w in vals.get("topic", "") for w in topics) else 0.75)
        hit += best
    extra = max(0, len(obj) - 3)
    return max(0, hit - 0.5 * extra) / 3, f"{hit:g}/3 meetings, {len(obj)} items"


def g_math(ans):
    v = to_num(last_tag(ans, "RESPUESTA"))
    return (1.0 if v is not None and abs(v - 47685) < 0.5 else 0.0), f"got {v}"


def g_logic(ans):
    v = last_tag(ans, "ANSWER") or last_tag(ans, "RESPUESTA") or ""
    return (1.0 if "carla" in norm(v) else 0.0), f"got {v[:30]}"


def g_date(ans):
    v = last_tag(ans, "ANSWER") or ""
    return (1.0 if "2026-04-27" in v else 0.0), f"got {v[:20]}"


def tests_grader(tests, fn_name):
    def g(ans):
        code = extract_code(ans)
        if fn_name not in code:
            return 0, "function missing"
        return run_py_tests(code, tests)
    return g


def g_email(ans):
    n = len(words(ans))
    return (0.5 if is_spanish(ans) else 0) + (0.5 if n <= 135 else 0), f"{n} words"


def g_post(ans):
    n = len(words(ans))
    emoji = bool(re.search(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", ans))
    return (0.4 if is_spanish(ans) else 0) + (0.4 if 70 <= n <= 135 else 0) + (0 if emoji else 0.2), f"{n} words, emoji={emoji}"


def g_5bullets_es(ans):
    lines = [l for l in ans.splitlines() if re.match(r"^\s*([-*•]|\d+[.)])\s", l)]
    return (0.6 if len(lines) == 5 else 0.2 if 4 <= len(lines) <= 6 else 0) + (0.4 if is_spanish(ans) else 0), f"{len(lines)} bullets"


def g_3sent_en(ans):
    sents = [s for s in re.split(r"(?<=[.!?])\s+", ans.strip()) if len(s.split()) > 2]
    return (0.6 if len(sents) == 3 else 0.2 if 2 <= len(sents) <= 4 else 0) + (0.4 if is_english(ans) else 0), f"{len(sents)} sentences"


def g_translate(ans):
    return (1.0 if is_spanish(ans) else 0.0), ""


# --- vision graders
def cer_score(ans, ref):
    a, r = norm(re.sub(r"\s+", " ", ans)).strip(), norm(re.sub(r"\s+", " ", ref)).strip()
    # Levenshtein
    prev = list(range(len(a) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, ac in enumerate(a, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != ac)))
        prev = cur
    cer = prev[-1] / max(1, len(r))
    return max(0.0, 1 - cer), f"CER {cer:.2%}"


def g_chart(ans):
    v = last_tag(ans, "RESPUESTA") or ans
    ok_c = "abril" in norm(v)
    ok_v = "73" in v
    return 0.5 * ok_c + 0.5 * ok_v, v[:40]


def g_ui(ans):
    labels = ["Subir archivo", "Ver comentarios", "Compartir enlace", "Eliminar proyecto"]
    hit = sum(norm(l) in norm(ans) for l in labels)
    return hit / 4, f"{hit}/4 labels"


def g_ground(ans):
    """Click target: 'Compartir enlace' button in ui_es.png (420x820), box x 24-396, y 628-692.
    Accepts pixel coords, 0-1000 normalized, or 0-1 normalized."""
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", ans)]
    if len(nums) < 2:
        return 0, "no coordinates"
    x, y = nums[0], nums[1]
    if len(nums) >= 4 and "[" in ans:  # bounding box: take its center
        x, y = (nums[0] + nums[2]) / 2, (nums[1] + nums[3]) / 2
    cands = [(x, y, "px"), (x * 420 / 1000, y * 820 / 1000, "0-1000"), (x * 420, y * 820, "0-1")]
    for cx, cy, mode in cands:
        if 24 - 8 <= cx <= 396 + 8 and 628 - 8 <= cy <= 692 + 8:
            return 1.0, f"hit ({mode}) {x:g},{y:g}"
    return 0.0, f"miss {x:g},{y:g}"


def g_shapes(ans):
    v = last_tag(ans, "RESPUESTA") or ans
    n = to_num(v)
    return (1.0 if n == 7 else 0.0), f"got {n}"


# ---------------------------------------------------------------- tool definitions
TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "create_event", "description": "Create a calendar event",
     "parameters": {"type": "object", "properties": {"title": {"type": "string"},
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                    "time": {"type": "string", "description": "HH:MM 24h"},
                    "duration_minutes": {"type": "integer"}}, "required": ["title", "date", "time"]}}},
    {"type": "function", "function": {"name": "search_notes", "description": "Search the user's notes vault",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"},
                    "limit": {"type": "integer"}}, "required": ["query"]}}},
]


def _calls(msg):
    out = []
    for c in (msg or {}).get("tool_calls") or []:
        f = c.get("function", {})
        try:
            args = json.loads(f.get("arguments") or "{}")
        except Exception:
            args = {"_raw": f.get("arguments")}
        out.append((f.get("name"), args))
    return out


def t_weather(msg):
    calls = _calls(msg)
    if not calls:
        return 0, "no tool call"
    name, a = calls[0]
    s = 0.5 * (name == "get_weather") + 0.3 * ("san nicolas" in norm(str(a.get("city", "")))) + 0.2 * (str(a.get("unit", "")).lower().startswith("c"))
    return s, f"{name} {json.dumps(a, ensure_ascii=False)[:80]}"


def t_event(msg):
    calls = _calls(msg)
    if not calls:
        return 0, "no tool call"
    name, a = calls[0]
    s = 0.25 * (name == "create_event") + 0.25 * (a.get("date") == "2026-11-12") + 0.25 * (str(a.get("time", "")).startswith("16:00")) + 0.25 * (to_num(str(a.get("duration_minutes"))) == 90)
    return s, f"{name} {json.dumps(a, ensure_ascii=False)[:100]}"


def t_abstain(msg):
    calls = _calls(msg)
    content = (msg or {}).get("content") or ""
    _, content = (content, content)
    return (0.5 if not calls else 0) + (0.5 if "391" in content else 0), ("called " + calls[0][0]) if calls else "no call"


# ---------------------------------------------------------------- task list

def tasks():
    mol = _read("passage_molinos.txt")
    cap1 = _read("passage_cap1.txt")
    T = []
    def add(id, cat, prompt, grader, kind="text", judge=None, max_tokens=16384, temperature=0.0, image=None, source=None):
        T.append(dict(id=id, cat=cat, prompt=prompt, grader=grader, kind=kind, judge=judge,
                      max_tokens=max_tokens, temperature=temperature, image=image, source=source))
    # instruction following
    add("if_bullets_es", "instructions", "Dame exactamente 4 viñetas (cada línea empieza con '- ') con consejos para un diseñador gráfico junior. Cada viñeta debe tener menos de 15 palabras. No escribas nada antes ni después de las viñetas.", g_bullets4)
    add("if_json_only", "instructions", "Return ONLY a JSON object, with no markdown and no explanation, with keys 'color' (a hex string like #1A2B3C), 'font' (a font name) and 'size' (an integer between 12 and 24) for a minimalist poster title.", g_json_only)
    add("if_lower_es", "instructions", "Escribe exactamente dos oraciones sobre la tipografía, todo en minúsculas, y termina el texto exactamente con la palabra 'fin.'", g_lower_fin)
    add("if_25words_en", "instructions", "Describe the Bauhaus movement in exactly 25 words. Output only those 25 words.", g_25words)
    add("if_forbidden_es", "instructions", "Explica qué es el kerning en un solo párrafo, sin usar las palabras 'espacio' ni 'letras'.", g_forbidden)
    # structured extraction
    add("json_factura_es", "json", "Extrae los datos de esta factura y responde SOLO con un objeto JSON con las claves: folio, fecha (formato YYYY-MM-DD), emisor, receptor, subtotal, iva, total (números sin símbolos).\n\nTexto: Factura F-2291 emitida el 3 de marzo de 2026 por Papelería Monterrey S.A. de C.V. a Estudio Rivas. Conceptos: 120 carteles a $35.50 cada uno y 2 rollos de vinil a $890.00 cada uno. Se aplica IVA del 16% sobre el subtotal.", g_factura)
    add("json_meetings_en", "json", "Extract every meeting from this email as a JSON list of objects with keys day, time (24h HH:MM) and topic (one lowercase word). Output only JSON.\n\nEmail: Hi team! Quick recap for next week. Tuesday at 9:30am we review the budget. On Wednesday we'll do the portfolio critique at 2pm. Thursday is free, enjoy. Friday 5:15pm is the launch rehearsal, don't be late.", g_meetings)
    # tool calling (non-streaming)
    add("tool_weather_es", "tools", "¿Qué temperatura hace ahora en San Nicolás de los Garza? La quiero en grados Celsius.", t_weather, kind="tool")
    add("tool_event_es", "tools", "Agenda una revisión de portafolio con mis alumnos el 12 de noviembre de 2026 a las 4 de la tarde. Dura hora y media.", t_event, kind="tool")
    add("tool_abstain", "tools", "¿Cuánto es 17 por 23? Responde solo con el resultado.", t_abstain, kind="tool")
    # coding
    add("code_slugify", "coding", "Escribe una función de Python `slugify(texto: str) -> str` que: convierta a minúsculas, quite acentos y diéresis (la á se vuelve a, la ñ se vuelve n, la ü se vuelve u), reemplace cualquier secuencia de caracteres no alfanuméricos por un solo guion y quite los guiones del inicio y del final. Responde solo con el código en un bloque ```python.",
        tests_grader(["slugify('¡Hola, Señor Ñandú!') == 'hola-senor-nandu'", "slugify('  Diseño   Gráfico 2026 ') == 'diseno-grafico-2026'", "slugify('pingüino--ágil') == 'pinguino-agil'", "slugify('---') == ''"], "slugify"))
    add("code_intervals", "coding", "Write a Python function `merge(intervals: list[list[int]]) -> list[list[int]]` that merges overlapping or touching closed intervals and returns them sorted. Reply with only the code in a ```python block.",
        tests_grader(["merge([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]", "merge([[1,4],[4,5]]) == [[1,5]]", "merge([]) == []", "merge([[5,7],[1,2]]) == [[1,2],[5,7]]"], "merge"))
    add("code_contrast", "coding", "Write a Python function `contrast_ratio(hex1: str, hex2: str) -> float` that returns the WCAG 2.x contrast ratio between two hex colors like '#777777', rounded to 2 decimals. Reply with only the code in a ```python block.",
        tests_grader(["contrast_ratio('#000000','#ffffff') == 21.0", "contrast_ratio('#ffffff','#000000') == 21.0", "abs(contrast_ratio('#777777','#ffffff') - 4.48) <= 0.01", "contrast_ratio('#123456','#123456') == 1.0"], "contrast_ratio"))
    # reasoning
    add("math_es", "reasoning", "Un estudio de diseño cobra $850 MXN por hora. Un proyecto dura 3 semanas, con 4 días de trabajo por semana y 5.5 horas por día. Al total se le aplica un descuento del 15%. ¿Cuánto se cobra en total? Termina con una línea con el formato 'RESPUESTA: <número>'.", g_math)
    add("logic_en", "reasoning", "Ana, Beto and Carla each designed exactly one of: a logo, a poster, a website. Ana did not design the website. Beto designed neither the logo nor the website. Who designed the website? End with a line 'ANSWER: <name>'.", g_logic)
    add("date_en", "reasoning", "A course starts on Monday, January 12, 2026 and has 16 weekly sessions, one every Monday with no breaks. On what date is the last session? End with a line 'ANSWER: YYYY-MM-DD'.", g_date)
    # writing (judged)
    add("write_email_es", "writing", "Escribe un correo breve (máximo 120 palabras) para un alumno que entregó tarde su proyecto final. Tono amable pero firme: acepta la entrega con 10% de penalización y ofrécele una asesoría el jueves a las 11.", g_email,
        judge="Evalúa el correo: ¿cumple todo lo pedido (acepta la entrega, 10% de penalización, asesoría el jueves a las 11)? ¿El tono es amable pero firme? ¿Suena natural, como lo escribiría un profesor mexicano y no un robot? ¿Está bien escrito en español?", temperature=0.7)
    add("write_post_es", "writing", "Escribe un párrafo para LinkedIn (de 80 a 120 palabras) sobre por qué los diseñadores deberían aprender a escribir bien. Que suene humano, sin clichés, sin hashtags y sin emojis.", g_post,
        judge="Evalúa el párrafo: ¿suena humano y natural, o genérico y como escrito por IA? ¿Evita clichés? ¿Tiene una idea clara y convincente? ¿Español correcto?", temperature=0.7)
    add("translate_es", "writing", "Translate this UI microcopy into natural Mexican Spanish for a design-school app. Output only the translation.\n\n\"Heads up! Your project is due in 2 days. Upload your files before Friday at noon so your instructor has time to review them. Need more time? Request an extension from the course page.\"", g_translate,
        judge="Evalúa la traducción: ¿conserva todo el significado (aviso, 2 días, subir archivos antes del viernes al mediodía, el profesor revisa, pedir prórroga desde la página del curso)? ¿Suena natural en español de México, como texto de interfaz?")
    add("summary_es", "summary", "Resume el siguiente texto en exactamente 5 viñetas en español. No inventes nada que no esté en el texto.\n\n" + mol, g_5bullets_es,
        judge="Evalúa el resumen contra el texto fuente: ¿es fiel (sin datos inventados)? ¿Cubre los hechos principales? ¿Son 5 viñetas claras?", source=mol)
    add("summary_en", "summary", "Summarize the following Spanish text in exactly 3 English sentences. Do not add anything that is not in the text.\n\n" + cap1, g_3sent_en,
        judge="Evaluate the English summary against the Spanish source: is it faithful (nothing invented)? Does it capture the main points? Is it exactly 3 clear sentences?", source=cap1)
    # vision
    ocr_ref = json.load(open(os.path.join(A, "images", "answers.json"), encoding="utf-8"))["ocr_es"]
    add("vis_ocr_es", "vision", "Transcribe exactamente el texto de la imagen. Responde solo con el texto.", lambda a: cer_score(a, ocr_ref), kind="vision", image="ocr_es.png", max_tokens=8192)
    add("vis_chart_es", "vision", "¿Qué mes tiene el valor más alto en la gráfica y cuál es ese valor? Termina con 'RESPUESTA: <mes>, <valor>'.", g_chart, kind="vision", image="chart_es.png", max_tokens=8192)
    add("vis_ui_es", "vision", "Lista el texto de cada botón que aparece en esta pantalla, uno por línea.", g_ui, kind="vision", image="ui_es.png", max_tokens=8192)
    add("vis_click_es", "vision", "La imagen mide 420x820 píxeles. Necesito hacer clic en el botón 'Compartir enlace'. Responde solo con las coordenadas del centro de ese botón, en píxeles, con el formato: x, y", g_ground, kind="vision", image="ui_es.png", max_tokens=8192)
    add("vis_shapes", "vision", "¿Cuántos triángulos hay en la imagen? Termina con 'RESPUESTA: <número>'.", g_shapes, kind="vision", image="shapes.png", max_tokens=8192)
    return T


JUDGE_SYSTEM = ("Eres un evaluador estricto y justo. Recibirás una tarea, (a veces) un texto fuente y una respuesta. "
                "Califica la respuesta del 1 al 10 según los criterios. Ignora la longitud excepto si incumple lo pedido. "
                "Responde con una línea 'PUNTAJE: <número del 1 al 10>' seguida de una sola frase de justificación.")


def judge_prompt(task, answer):
    src = f"\n\nTEXTO FUENTE:\n{task['source']}" if task.get("source") else ""
    return (f"TAREA:\n{task['prompt'] if not task.get('source') else task['prompt'].split(chr(10)+chr(10))[0]}{src}\n\n"
            f"CRITERIOS:\n{task['judge']}\n\nRESPUESTA A EVALUAR:\n{answer}\n\n"
            "Recuerda: responde con 'PUNTAJE: <1-10>' y una frase.")


def parse_judge(text):
    m = re.findall(r"(?:PUNTAJE|SCORE)\s*[:=]\s*(\d+(?:\.\d+)?)", text or "", re.I)
    if not m:
        return None
    v = float(m[-1])
    return v if 1 <= v <= 10 else None
