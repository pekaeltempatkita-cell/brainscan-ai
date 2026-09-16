"""
report_pdf.py — Generate laporan hasil skrining dalam format PDF,
supaya bisa diunduh dokter/pasien (tombol "Unduh Laporan").

Pakai reportlab murni (tanpa wkhtmltopdf/weasyprint) biar gampang di-deploy
di server mana pun tanpa dependency sistem tambahan.
"""
import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable,
)

from config import APP_NAME, FIGURES_DIR

# Nama tampilan yang lebih manusiawi untuk tiap label kelas
LABEL_DISPLAY = {
    "Alzheimer_Mild": "Alzheimer (Ringan)",
    "Alzheimer_Moderate": "Alzheimer (Sedang)",
    "Alzheimer_Very_Mild": "Alzheimer (Sangat Ringan)",
    "Intracranial_Hemorrhage": "Pendarahan Intrakranial (Intracranial Hemorrhage)",
    "Multiple_Sclerosis": "Multiple Sclerosis",
    "Normal_Healthy": "Normal / Sehat",
    "Stroke_Iskemik": "Stroke Iskemik",
    "Tumor_Glioma": "Tumor Otak — Glioma",
    "Tumor_Meningioma": "Tumor Otak — Meningioma",
    "Tumor_Pituitary": "Tumor Otak — Pituitary",
}

RECOMMENDED_SPECIALIST = {
    "Alzheimer_Mild": "Sp.N (Neurologi) — sub-spesialisasi neurodegeneratif/memori",
    "Alzheimer_Moderate": "Sp.N (Neurologi) — sub-spesialisasi neurodegeneratif/memori",
    "Alzheimer_Very_Mild": "Sp.N (Neurologi) — sub-spesialisasi neurodegeneratif/memori",
    "Intracranial_Hemorrhage": "Sp.BS (Bedah Saraf) SEGERA/CITO + Sp.N (Neurologi)",
    "Multiple_Sclerosis": "Sp.N (Neurologi) — sub-spesialisasi neuroimunologi",
    "Normal_Healthy": "Tidak ada indikasi khusus; kontrol rutin sesuai anjuran dokter",
    "Stroke_Iskemik": "Sp.N (Neurologi) SEGERA + evaluasi unit stroke",
    "Tumor_Glioma": "Sp.BS (Bedah Saraf) + Sp.Onk.Rad (Onkologi Radiasi)",
    "Tumor_Meningioma": "Sp.BS (Bedah Saraf)",
    "Tumor_Pituitary": "Sp.BS (Bedah Saraf) + Sp.PD-KEMD (Endokrinologi)",
}


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle(name="AppTitle", fontSize=17, leading=20, textColor=colors.HexColor("#0b1c3f"), fontName="Helvetica-Bold"))
    ss.add(ParagraphStyle(name="SectionHead", fontSize=11.5, leading=14, spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#0b1c3f"), fontName="Helvetica-Bold"))
    ss.add(ParagraphStyle(name="Body", fontSize=9.5, leading=14, textColor=colors.HexColor("#1f2937")))
    ss.add(ParagraphStyle(name="Muted", fontSize=8.5, leading=12, textColor=colors.HexColor("#5b6b85")))
    ss.add(ParagraphStyle(name="Disclaimer", fontSize=8, leading=11, textColor=colors.HexColor("#5b6b85"), borderPadding=8))
    return ss


def generate_report_pdf(patient: dict, prediction: dict, record: dict) -> bytes:
    """
    patient    : dict dari database.get_patient_by_nik()
    prediction : dict dari database.get_prediction_by_id()
    record     : dict 1 baris medical_records (buat catatan_dokter & created_at rekam medis)
    Return     : bytes isi file PDF (siap dikirim sebagai response download).
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=18 * mm, bottomMargin=16 * mm, leftMargin=18 * mm, rightMargin=18 * mm,
    )
    ss = _styles()
    story = []

    # --- Header ---
    story.append(Paragraph(f"{APP_NAME}", ss["AppTitle"]))
    story.append(Paragraph("Laporan Hasil Skrining AI — Citra MRI/CT Otak", ss["Muted"]))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#2f6fed"), thickness=1.4))
    story.append(Spacer(1, 10))

    # --- Identitas pasien ---
    story.append(Paragraph("I. IDENTITAS PASIEN", ss["SectionHead"]))
    identity_data = [
        ["Nama Pasien", patient.get("nama") or "-", "NIK", patient.get("nik") or "-"],
        ["Tanggal Lahir", patient.get("tanggal_lahir") or "-", "Jenis Kelamin", patient.get("jenis_kelamin") or "-"],
        ["Alamat", patient.get("alamat") or "-", "No. Telepon", patient.get("no_telepon") or "-"],
    ]
    t = Table(identity_data, colWidths=[32 * mm, 60 * mm, 32 * mm, 46 * mm])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1f2937")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor("#e2e8f5")),
    ]))
    story.append(t)

    # --- Info pemeriksaan ---
    story.append(Paragraph("II. INFORMASI PEMERIKSAAN", ss["SectionHead"]))
    upload_time = prediction.get("upload_time", "")
    story.append(Paragraph(f"Tanggal Analisis: {upload_time}", ss["Body"]))
    story.append(Paragraph(f"File Citra: {prediction.get('filename', '-')}", ss["Body"]))
    story.append(Paragraph(
        f"Precheck kualitas citra (deteksi apakah citra otak): "
        f"<b>{'Valid — citra otak terkonfirmasi' if prediction.get('precheck_status') == 'valid' else 'Tidak valid'}</b> "
        f"(confidence {prediction.get('precheck_confidence', 0) * 100:.1f}%)",
        ss["Body"]))

    # --- Hasil klasifikasi ---
    label = prediction.get("prediction_label")
    conf = prediction.get("prediction_confidence") or 0
    story.append(Paragraph("III. HASIL KLASIFIKASI AI (OPINI SEKUNDER)", ss["SectionHead"]))
    if label:
        display_label = LABEL_DISPLAY.get(label, label)
        story.append(Paragraph(f"<b>Prediksi Utama: {display_label}</b>", ss["Body"]))
        story.append(Paragraph(f"Tingkat keyakinan model (confidence): <b>{conf * 100:.2f}%</b>", ss["Body"]))
        story.append(Spacer(1, 6))

        # Tabel seluruh probabilitas per kelas, diurutkan dari yang tertinggi
        import json as _json
        all_probs = prediction.get("all_probabilities")
        if all_probs:
            if isinstance(all_probs, str):
                all_probs = _json.loads(all_probs)
            sorted_probs = sorted(all_probs.items(), key=lambda kv: kv[1], reverse=True)
            rows = [["Kelas", "Probabilitas"]] + [
                [LABEL_DISPLAY.get(k, k), f"{v * 100:.2f}%"] for k, v in sorted_probs
            ]
            prob_table = Table(rows, colWidths=[110 * mm, 40 * mm])
            prob_table.setStyle(TableStyle([
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3fb")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f5")),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#fff7e6")),  # highlight baris teratas
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(prob_table)
        story.append(Spacer(1, 8))

        # --- Rekomendasi ---
        story.append(Paragraph("IV. REKOMENDASI TINDAK LANJUT", ss["SectionHead"]))
        rec = RECOMMENDED_SPECIALIST.get(label, "Konsultasikan dengan dokter spesialis terkait.")
        story.append(Paragraph(f"Saran konsultasi: <b>{rec}</b>", ss["Body"]))

        # --- Penjelasan Gemini (kalau ada) ---
        explanation = prediction.get("gemini_explanation")
        story.append(Paragraph("V. PENJELASAN EDUKASI (AI)", ss["SectionHead"]))
        story.append(Paragraph(
            explanation.replace("\n", "<br/>") if explanation
            else "Penjelasan otomatis belum tersedia untuk hasil ini.",
            ss["Body"]))
        # --- Grad-CAM ---
        gradcam_path = prediction.get("gradcam_path")
        if gradcam_path:
            full_path = FIGURES_DIR.parent / gradcam_path if not str(gradcam_path).startswith(str(FIGURES_DIR)) else gradcam_path
            try:
                story.append(Paragraph("VI. PETA ATENSI MODEL (EXPLAINABLE AI / GRAD-CAM)", ss["SectionHead"]))
                story.append(Paragraph(
                    "Area yang diwarnai merah/kuning menunjukkan region citra yang paling "
                    "mempengaruhi keputusan model. Ini alat bantu interpretasi, bukan diagnosis.",
                    ss["Muted"]))
                story.append(Spacer(1, 4))
                story.append(Image(str(full_path), width=70 * mm, height=70 * mm))
            except Exception:
                pass  # kalau file gambar gak ketemu/rusak, skip aja, jangan gagalkan seluruh PDF
    else:
        story.append(Paragraph(
            "Citra yang diunggah tidak terdeteksi sebagai citra MRI/CT otak yang valid, "
            "sehingga klasifikasi penyakit tidak dapat dilakukan.", ss["Body"]))

    # --- Catatan dokter (kalau ada di rekam medis) ---
    if record and record.get("catatan_dokter"):
        story.append(Paragraph("VII. CATATAN DOKTER", ss["SectionHead"]))
        story.append(Paragraph(record["catatan_dokter"].replace("\n", "<br/>"), ss["Body"]))

    # --- Disclaimer ---
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#e2e8f5"), thickness=0.8))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Catatan Penting:</b> Laporan ini dihasilkan oleh sistem kecerdasan buatan (AI) sebagai "
        "alat bantu skrining awal (decision support tool), BUKAN diagnosis medis final. Hasil ini "
        "wajib dikonfirmasi, dikorelasikan secara klinis, dan divalidasi oleh Dokter Spesialis "
        "Radiologi (Sp.Rad) atau dokter spesialis terkait sebelum digunakan sebagai dasar tindakan medis.",
        ss["Disclaimer"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"Dokumen dibuat otomatis oleh sistem pada {datetime.now().strftime('%d %B %Y, %H:%M')}.", ss["Muted"]))

    doc.build(story)
    return buf.getvalue()
