document.addEventListener("click", function (event) {
  const printButton = event.target && event.target.closest(".btn-print, .btn-report-print");
  if (printButton) {
    event.preventDefault();
    if (printButton.classList.contains("btn-report-print")) {
      document.body.classList.add("report-print-mode");
    }
    window.print();
  }
});

window.addEventListener("afterprint", function () {
  document.body.classList.remove("report-print-mode");
});
