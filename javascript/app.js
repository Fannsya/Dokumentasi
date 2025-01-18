function TambahParagraf() {
    const para = document.createElement("p");
    para.textContent = "Selamat anda berhasil donasi"
    document.body.appendChild(para);
}

const buttons = document.querySelectorAll("button");

for (const button of buttons) {
    button.addEventListener("click",
        TambahParagraf);
}