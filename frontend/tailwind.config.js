/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}"
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["'Work Sans'", "system-ui", "sans-serif"],
      },
      colors: {
        mint: {
          50: "#F2FBF8",
          100: "#DDF6EF",
          200: "#B9ECDD",
          300: "#86DDC6",
          400: "#4EC8AA",
          500: "#27A98D",
        },
        lilac: {
          50: "#F7F2FF",
          100: "#EBDDFF",
          200: "#D5B8FF",
          300: "#BD8CFF",
          400: "#A25BFF",
          500: "#8736E6",
        },
        sky: {
          50: "#F2F9FF",
          100: "#DDF0FF",
          200: "#B5DEFF",
          300: "#7CC4FF",
          400: "#3BA3FF",
          500: "#167FE6",
        },
      },
    },
  },
  plugins: [],
};
