# -*- coding: utf-8 -*-
"""Build the bachelor qualification thesis (.docx) to KNU FCSC formatting spec.

Run from the repository root:  python3 thesis/build_thesis.py
Output: thesis/Кваліфікаційна_робота_Макаренко.docx

Formatting follows the faculty methodology (A4, Times New Roman 14, line
spacing 1.5, margins L25/R10/T20/B20 mm, indent 1.27 cm, page numbers
top-right with no number on the title page, ДСТУ 8302:2015 references).
Numeric results are taken verbatim from runs/summary.md / runs/THESIS_RESULTS.md.
"""
from __future__ import annotations

import os

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNF = os.path.join(ROOT, "runs", "figures")
TFIG = os.path.join(ROOT, "thesis", "figs")
OUT = os.path.join(ROOT, "thesis", "Кваліфікаційна_робота_Макаренко.docx")

FONT = "Times New Roman"
SIZE = 14

# running figure/table counters reset per chapter
_state = {"chapter": 0, "fig": 0, "tab": 0}


# --------------------------------------------------------------------- helpers
def set_cell_font(cell, bold=False, size=12, align=WD_ALIGN_PARAGRAPH.CENTER):
    for p in cell.paragraphs:
        p.alignment = align
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.first_line_indent = Cm(0)
        for r in p.runs:
            r.font.name = FONT
            r.font.size = Pt(size)
            r.font.bold = bold
            r._element.rPr.rFonts.set(qn("w:cs"), FONT)


def _field(paragraph, instr):
    """Insert a Word field (e.g. PAGE, TOC) into a paragraph."""
    run = paragraph.add_run()
    fldBegin = OxmlElement("w:fldChar"); fldBegin.set(qn("w:fldCharType"), "begin")
    instrText = OxmlElement("w:instrText"); instrText.set(qn("xml:space"), "preserve")
    instrText.text = instr
    fldSep = OxmlElement("w:fldChar"); fldSep.set(qn("w:fldCharType"), "separate")
    fldEnd = OxmlElement("w:fldChar"); fldEnd.set(qn("w:fldCharType"), "end")
    run._r.append(fldBegin); run._r.append(instrText); run._r.append(fldSep); run._r.append(fldEnd)
    return run


class Builder:
    def __init__(self):
        self.doc = Document()
        self._base_style()
        self._heading_styles()

    # ---- styling -----------------------------------------------------------
    def _base_style(self):
        st = self.doc.styles["Normal"]
        st.font.name = FONT
        st.font.size = Pt(SIZE)
        st.element.rPr.rFonts.set(qn("w:cs"), FONT)
        pf = st.paragraph_format
        pf.line_spacing = 1.5
        pf.space_after = Pt(0)
        pf.space_before = Pt(0)
        pf.first_line_indent = Cm(1.27)
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

        sec = self.doc.sections[0]
        sec.page_height = Cm(29.7)
        sec.page_width = Cm(21.0)
        sec.top_margin = Cm(2.0)
        sec.bottom_margin = Cm(2.0)
        sec.left_margin = Cm(2.5)
        sec.right_margin = Cm(1.0)

    def _heading_styles(self):
        for name, size, bold in (("Heading 1", SIZE, True),
                                 ("Heading 2", SIZE, True),
                                 ("Heading 3", SIZE, True)):
            s = self.doc.styles[name]
            s.font.name = FONT
            s.font.size = Pt(size)
            s.font.bold = bold
            s.font.color.rgb = RGBColor(0, 0, 0)
            s.element.rPr.rFonts.set(qn("w:cs"), FONT)
            pf = s.paragraph_format
            pf.line_spacing = 1.5
            pf.keep_with_next = True
            pf.space_before = Pt(0)
            pf.space_after = Pt(12)

    # ---- header / numbering ------------------------------------------------
    def enable_page_numbers(self):
        sec = self.doc.sections[0]
        sec.different_first_page_header_footer = True
        # first page (title) header empty -> no number
        sec.first_page_header.is_linked_to_previous = False
        # default header: page number top-right
        hdr = sec.header
        hdr.is_linked_to_previous = False
        p = hdr.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p.paragraph_format.first_line_indent = Cm(0)
        r = _field(p, "PAGE")
        for rr in p.runs:
            rr.font.name = FONT
            rr.font.size = Pt(SIZE)

    # ---- content primitives ------------------------------------------------
    def para(self, text, align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=True,
             bold=False, italic=False, size=SIZE, space_after=0, center=False):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else align
        pf = p.paragraph_format
        pf.first_line_indent = Cm(1.27 if indent else 0)
        pf.line_spacing = 1.5
        pf.space_after = Pt(space_after)
        r = p.add_run(text)
        r.font.name = FONT
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r._element.rPr.rFonts.set(qn("w:cs"), FONT)
        return p

    def bullets(self, items, dash=True):
        for it in items:
            p = self.doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            pf = p.paragraph_format
            pf.first_line_indent = Cm(1.27)
            pf.line_spacing = 1.5
            pf.space_after = Pt(0)
            r = p.add_run(("– " if dash else "") + it)
            r.font.name = FONT; r.font.size = Pt(SIZE)
            r._element.rPr.rFonts.set(qn("w:cs"), FONT)

    def h1(self, text, numbered=False):
        _state["fig"] = 0; _state["tab"] = 0
        if numbered:
            _state["chapter"] += 1
            text = f"{_state['chapter']} {text}"
        h = self.doc.add_heading(level=1)
        h.alignment = WD_ALIGN_PARAGRAPH.CENTER
        h.paragraph_format.page_break_before = True
        h.paragraph_format.first_line_indent = Cm(0)
        r = h.add_run(text.upper())
        r.font.name = FONT; r.font.size = Pt(SIZE); r.font.bold = True
        r.font.color.rgb = RGBColor(0, 0, 0)
        r._element.rPr.rFonts.set(qn("w:cs"), FONT)
        return h

    def h2(self, num, text):
        h = self.doc.add_heading(level=2)
        h.alignment = WD_ALIGN_PARAGRAPH.LEFT
        h.paragraph_format.first_line_indent = Cm(1.27)
        h.paragraph_format.space_before = Pt(6)
        r = h.add_run(f"{num} {text}")
        r.font.name = FONT; r.font.size = Pt(SIZE); r.font.bold = True
        r.font.color.rgb = RGBColor(0, 0, 0)
        r._element.rPr.rFonts.set(qn("w:cs"), FONT)
        return h

    def figure(self, path, caption, width_cm=15.0):
        _state["fig"] += 1
        num = f"{_state['chapter']}.{_state['fig']}"
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.keep_with_next = True
        run = p.add_run()
        run.add_picture(path, width=Cm(width_cm))
        cap = self.doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.first_line_indent = Cm(0)
        cap.paragraph_format.space_after = Pt(8)
        r = cap.add_run(f"Рисунок {num} — {caption}")
        r.font.name = FONT; r.font.size = Pt(SIZE)
        r._element.rPr.rFonts.set(qn("w:cs"), FONT)

    def table(self, caption, headers, rows, widths=None, first_col_left=True,
              header_size=11, body_size=11):
        _state["tab"] += 1
        num = f"{_state['chapter']}.{_state['tab']}"
        cap = self.doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.LEFT
        cap.paragraph_format.first_line_indent = Cm(1.27)
        cap.paragraph_format.space_before = Pt(6)
        cap.paragraph_format.keep_with_next = True
        r = cap.add_run(f"Таблиця {num} — {caption}")
        r.font.name = FONT; r.font.size = Pt(SIZE)
        r._element.rPr.rFonts.set(qn("w:cs"), FONT)

        t = self.doc.add_table(rows=1, cols=len(headers))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        hcells = t.rows[0].cells
        for i, htext in enumerate(headers):
            hcells[i].text = htext
            set_cell_font(hcells[i], bold=True, size=header_size)
        for row in rows:
            cells = t.add_row().cells
            for i, val in enumerate(row):
                cells[i].text = str(val)
                align = (WD_ALIGN_PARAGRAPH.LEFT if (i == 0 and first_col_left)
                         else WD_ALIGN_PARAGRAPH.CENTER)
                set_cell_font(cells[i], bold=False, size=body_size, align=align)
        if widths:
            for i, w in enumerate(widths):
                for row in t.rows:
                    row.cells[i].width = Cm(w)
        return t

    def lead_para(self, label, text):
        """Paragraph that opens with a bold lead-in label (as in the вступ example)."""
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf = p.paragraph_format
        pf.first_line_indent = Cm(1.27)
        pf.line_spacing = 1.5
        pf.space_after = Pt(0)
        r1 = p.add_run(label + " ")
        r1.font.name = FONT; r1.font.size = Pt(SIZE); r1.font.bold = True
        r1._element.rPr.rFonts.set(qn("w:cs"), FONT)
        r2 = p.add_run(text)
        r2.font.name = FONT; r2.font.size = Pt(SIZE)
        r2._element.rPr.rFonts.set(qn("w:cs"), FONT)
        return p

    def set_chapter(self, label):
        _state["chapter"] = label
        _state["fig"] = 0
        _state["tab"] = 0

    def pagebreak(self):
        self.doc.add_page_break()

    def save(self):
        self.doc.save(OUT)


# ============================================================ document content
def build():
    b = Builder()
    d = b.doc

    # ----------------------------------------------------------- TITLE PAGE
    def tp(text, bold=False, size=14, center=True, before=0, after=0, indent=False):
        p = d.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT
        pf = p.paragraph_format
        pf.first_line_indent = Cm(0)
        pf.line_spacing = 1.5
        pf.space_before = Pt(before); pf.space_after = Pt(after)
        r = p.add_run(text)
        r.font.name = FONT; r.font.size = Pt(size); r.font.bold = bold
        r._element.rPr.rFonts.set(qn("w:cs"), FONT)
        return p

    tp("Київський національний університет імені Тараса Шевченка".upper(), bold=True)
    tp("Факультет комп’ютерних наук та кібернетики", size=14)
    tp("Кафедра теоретичної кібернетики", size=14, after=24)
    tp("Кваліфікаційна робота", size=14)
    tp("на здобуття ступеня бакалавра", size=14)
    tp("за спеціальністю 122 Комп’ютерні науки", size=14, after=18)
    tp("на тему:", size=14)
    tp("УНІФІКОВАНА СИСТЕМА ОБРОБКИ ТА КЛАСИФІКАЦІЇ ЕКГ-СИГНАЛІВ "
       "РІЗНИХ ФОРМАТІВ З ВИКОРИСТАННЯМ МЕТОДІВ МАШИННОГО НАВЧАННЯ",
       bold=True, size=14, after=24)

    # performer / advisor block, right-aligned
    def rblock(lines, after=12):
        p = d.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        pf = p.paragraph_format
        pf.first_line_indent = Cm(0); pf.left_indent = Cm(8.0)
        pf.line_spacing = 1.2; pf.space_after = Pt(after)
        for i, ln in enumerate(lines):
            r = p.add_run(("\n" if i else "") + ln)
            r.font.name = FONT; r.font.size = Pt(13)
            r._element.rPr.rFonts.set(qn("w:cs"), FONT)
        return p

    rblock(["Виконав студент 4-го курсу",
            "ОПП «Інформатика»",
            "групи ТК-41",
            "Макаренко Вадим Анатолійович",
            "______________ (підпис)"])
    rblock(["Науковий керівник:",
            "завідувач кафедри теоретичної кібернетики,",
            "доктор фізико-математичних наук, професор,",
            "член-кореспондент НАН України",
            "Крак Юрій Васильович",
            "______________ (підпис)"])
    rblock(["Засвідчую, що в цій роботі немає запозичень",
            "з праць інших авторів без відповідних посилань.",
            "Студент ______________ (підпис)"])

    tp("Роботу розглянуто й допущено до захисту на засіданні", size=12, before=6)
    tp("кафедри теоретичної кібернетики", size=12)
    tp("«____» ______________ 2026 р., протокол № ____", size=12)
    tp("Завідувач кафедри теоретичної кібернетики", size=12, before=6)
    tp("доктор фізико-математичних наук, професор Юрій КРАК   ______________", size=12)
    tp("(підпис)", size=11, after=12)
    tp("КИЇВ 2026", bold=True, size=14, before=18)

    b.enable_page_numbers()

    # ------------------------------------------------------------- РЕФЕРАТ
    content_referat(b)
    content_abbr(b)
    content_toc(b)
    content_intro(b)
    content_ch1(b)
    content_ch2(b)
    content_ch3(b)
    content_ch4(b)
    content_conclusions(b)
    content_references(b)
    content_appendix(b)

    b.save()
    print("Saved:", OUT)


# ---- the long-form chapters live in thesis_content.py to keep this readable
from thesis_content import (content_abbr, content_appendix, content_ch1,
                            content_ch2, content_ch3, content_ch4,
                            content_conclusions, content_intro,
                            content_referat, content_references, content_toc)

if __name__ == "__main__":
    build()
