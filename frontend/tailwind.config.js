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
      fontSize: {
        // Fluid type scale — clamp(min, preferred-vw, max)
        "fluid-2xs": "clamp(11px, 1.0vw,  13px)",
        "fluid-xs":  "clamp(12px, 1.1vw,  14px)",
        "fluid-sm":  "clamp(13px, 1.2vw,  15px)",
        "fluid-base":"clamp(14px, 1.3vw,  17px)",
        "fluid-lg":  "clamp(16px, 1.5vw,  20px)",
        "fluid-xl":  "clamp(18px, 1.8vw,  24px)",
        "fluid-2xl": "clamp(22px, 2.2vw,  30px)",
        "fluid-3xl": "clamp(26px, 2.8vw,  38px)",
        "fluid-4xl": "clamp(32px, 3.5vw,  52px)",
      },
      maxWidth: {
        // ~80 vw but capped at 1400 px
        "8xl": "min(80vw, 1400px)",
      },
      colors: {
        brand: {
          50:  "#f0f7ff",
          100: "#e0efff",
          200: "#b9dfff",
          300: "#7cc5ff",
          400: "#36a8ff",
          500: "#0c8ce9",
          600: "#006ec7",
          700: "#0058a1",
          800: "#034b85",
          900: "#083f6e",
        },
        pastel: {
          blue:   "#e8f4fd",
          mint:   "#e6f7f2",
          peach:  "#fef0e8",
          lilac:  "#f0eaf8",
          rose:   "#fce8ee",
          sky:    "#dbeafe",
          cream:  "#fefce8",
        },
        medical: {
          teal:   "#0d9488",
          cyan:   "#06b6d4",
          blue:   "#3b82f6",
          green:  "#10b981",
          red:    "#ef4444",
          orange: "#f59e0b",
        },
        surface: {
          0:   "#ffffff",
          50:  "#f8fafc",
          100: "#f1f5f9",
          200: "#e2e8f0",
          300: "#cbd5e1",
        },
      },
      boxShadow: {
        card:        "0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03)",
        "card-hover":"0 4px 16px rgba(0,0,0,0.06), 0 8px 32px rgba(0,0,0,0.04)",
        elevated:    "0 8px 30px rgba(0,0,0,0.06)",
        brand:       "0 4px 20px rgba(12,140,233,0.2)",
        "brand-lg":  "0 8px 32px rgba(12,140,233,0.25)",
        "glow-green":"0 4px 20px rgba(16,185,129,0.2)",
        "glow-red":  "0 4px 20px rgba(239,68,68,0.15)",
      },
      animation: {
        "fade-in-up": "fadeInUp 0.5s ease-out forwards",
        "fade-in":    "fadeIn 0.4s ease-out forwards",
        "slide-up":   "slideUp 0.5s cubic-bezier(0.16,1,0.3,1) forwards",
        "pulse-soft": "pulseSoft 3s ease-in-out infinite",
        shimmer:      "shimmer 2.5s linear infinite",
        gauge:        "gauge 1.2s ease-out forwards",
        heartbeat:    "heartbeat 1.5s ease-in-out infinite",
        float:        "float 6s ease-in-out infinite",
        "float-slow": "float 9s ease-in-out infinite",
        "ecg-trace":  "ecgTrace 2.5s ease-in-out forwards",
        "scale-in":   "scaleIn 0.3s ease-out forwards",
      },
      keyframes: {
        fadeInUp: {
          from: { opacity: "0", transform: "translateY(16px)" },
          to:   { opacity: "1", transform: "translateY(0)" },
        },
        fadeIn: {
          from: { opacity: "0" },
          to:   { opacity: "1" },
        },
        slideUp: {
          from: { opacity: "0", transform: "translateY(24px)" },
          to:   { opacity: "1", transform: "translateY(0)" },
        },
        pulseSoft: {
          "0%, 100%": { opacity: "1" },
          "50%":      { opacity: "0.6" },
        },
        shimmer: {
          "0%":   { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        gauge: {
          from: { strokeDashoffset: "301.59" },
        },
        heartbeat: {
          "0%, 100%": { transform: "scale(1)" },
          "14%":      { transform: "scale(1.15)" },
          "28%":      { transform: "scale(1)" },
          "42%":      { transform: "scale(1.1)" },
          "56%":      { transform: "scale(1)" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%":      { transform: "translateY(-12px)" },
        },
        ecgTrace: {
          from: { strokeDashoffset: "800" },
          to:   { strokeDashoffset: "0" },
        },
        scaleIn: {
          from: { opacity: "0", transform: "scale(0.95)" },
          to:   { opacity: "1", transform: "scale(1)" },
        },
      },
    },
  },
  plugins: [],
};
