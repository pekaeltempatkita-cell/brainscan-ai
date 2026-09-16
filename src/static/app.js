// app.js — interaksi kecil: drag & drop upload box + preview nama file.
// Tidak ada logic prediksi di sini -- semua inferensi tetap terjadi di server (app.py).

document.addEventListener("DOMContentLoaded", () => {
    const dropzone = document.querySelector("[data-dropzone]");
    if (!dropzone) return;

    const input = dropzone.querySelector("input[type=file]");
    const filenameEl = dropzone.querySelector("[data-filename]");

    const showFilename = (file) => {
        if (file && filenameEl) {
            filenameEl.textContent = "File dipilih: " + file.name;
            filenameEl.style.display = "block";
        }
    };

    dropzone.addEventListener("click", () => input && input.click());

    input && input.addEventListener("change", () => {
        if (input.files && input.files[0]) showFilename(input.files[0]);
    });

    ["dragenter", "dragover"].forEach((evt) => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach((evt) => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("dragover");
        });
    });

    dropzone.addEventListener("drop", (e) => {
        const file = e.dataTransfer.files && e.dataTransfer.files[0];
        if (file && input) {
            input.files = e.dataTransfer.files;
            showFilename(file);
        }
    });
});

// Isi lebar confidence-bar dari data-attribute (bukan langsung inline style="width:{{ }}%")
// biar linter CSS di VS Code gak salah baca sintaks Jinja sebagai CSS.
document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("[data-confidence-width]").forEach((el) => {
        const pct = parseFloat(el.getAttribute("data-confidence-width")) || 0;
        el.style.width = pct + "%";
    });
});