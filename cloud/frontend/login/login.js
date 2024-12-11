async function login() {
    const email = document.getElementById("email").value;
    const password = document.getElementById("password").value;
    const username = document.getElementById("username").value;

    try {
        const response = await fetch("http://localhost:5000/login", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email, password, username })
        });

        const data = await response.json();

        if (response.ok) {
            // Redirect to the admin page if login is successful
            window.location.href = data.redirect_url;
        } else {
            // Show an error message and reload the page
            alert(data.error || "Login failed. Please try again.");
            window.location.reload(); // Reload the page
        }
    } catch (error) {
        console.error("Error during login:", error);
        alert("An error occurred. Please try again.");
        window.location.reload(); // Reload the page
    }
}
