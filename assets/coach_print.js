document.addEventListener("click", function (event) {
  if (event.target && event.target.classList.contains("btn-print")) {
    window.print();
  }
});
