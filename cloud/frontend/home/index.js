// JavaScript to control section display
function enterSite() {
    document.querySelector('.landing').style.display = 'none';
    document.getElementById('main-content').style.display = 'block';
}

function showSection(sectionId) {
    document.querySelectorAll('.section-content').forEach(section => {
        section.style.display = 'none';
    });
    document.getElementById(sectionId).style.display = 'block';
}

function closeSection() {
    document.querySelectorAll('.section-content').forEach(section => {
        section.style.display = 'none';
    });
}