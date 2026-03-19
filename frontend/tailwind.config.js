/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}"
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["'Manrope'", "system-ui", "sans-serif"],
      },
      colors: {
        brand: {
          50:  "#eef6ff",
          100: "#d9ebff",
          200: "#bcdcff",
          300: "#8ec7ff",
          400: "#59a7ff",
          500: "#3383fc",
          600: "#1d64f1",
          700: "#154fde",
          800: "#1840b4",
          900: "#1a398e",
        },
        medical: {
          teal:   "#0d9488",
          cyan:   "#06b6d4",
          navy:   "#0f172a",
          deep:   "#0a0e27",
          card:   "rgba(255,255,255,0.06)",
          border: "rgba(255,255,255,0.10)",
        },
      },
      animation: {
        "float":      "float 6s ease-in-out infinite",
        "float-slow": "float 8s ease-in-out infinite",
        "fade-in-up": "fadeInUp 0.6s ease-out forwards",
        "pulse-glow": "pulseGlow 2s ease-in-out infinite",
        "shimmer":    "shimmer 2s linear infinite",
        "gauge":      "gauge 1.2s ease-out forwards",
      },
      keyframes: {
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%":      { transform: "translateY(-20px)" },
        },
        fadeInUp: {
          from: { opacity: "0", transform: "translateY(20px)" },
          to:   { opacity: "1", transform: "translateY(0)" },
        },
        pulseGlow: {
          "0%, 100%": { opacity: "1" },
          "50%":      { opacity: "0.5" },
        },
        shimmer: {
          "0%":   { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        gauge: {
          from: { strokeDashoffset: "301.59" },
        },
      },
    },
  },
  plugins: [],
};
