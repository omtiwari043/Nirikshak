/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#edf3fa", 100: "#d9e6f4", 500: "#123f76", 600: "#0b3264", 700: "#082a56", 900: "#041c43",
        },
      },
    },
  },
  plugins: [],
}
