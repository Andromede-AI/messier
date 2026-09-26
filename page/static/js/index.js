$(document).ready(function () {
  $('.navbar-burger').click(function () {
    $('.navbar-burger').toggleClass('is-active');
    $('.navbar-menu').toggleClass('is-active');
  });

  const copyButton = document.getElementById('copy-bibtex');
  let copyResetTimer;
  copyButton.addEventListener('click', async function () {
    const citation = document.getElementById('bibtex-code').textContent.trim();
    await navigator.clipboard.writeText(citation);

    copyButton.classList.add('is-copied');
    copyButton.title = 'Copied';
    copyButton.setAttribute('aria-label', 'Copied');
    clearTimeout(copyResetTimer);
    copyResetTimer = setTimeout(function () {
      copyButton.classList.remove('is-copied');
      copyButton.title = 'Copy BibTeX';
      copyButton.setAttribute('aria-label', 'Copy BibTeX');
    }, 1800);
  });
});
