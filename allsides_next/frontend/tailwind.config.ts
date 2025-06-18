import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "rgb(var(--gray-200) / <alpha-value>)",
        foreground: "rgb(var(--gray-900) / <alpha-value>)",
        primary: {
          DEFAULT: "rgb(var(--primary) / <alpha-value>)",
          light: "rgb(var(--primary-light) / <alpha-value>)",
          dark: "rgb(var(--primary-dark) / <alpha-value>)",
        },
        accent: {
          purple: "rgb(var(--accent-purple) / <alpha-value>)",
          rose: "rgb(var(--accent-rose) / <alpha-value>)",
          amber: "rgb(var(--accent-amber) / <alpha-value>)",
        },
        gray: {
          50: "rgb(var(--gray-50) / <alpha-value>)",
          100: "rgb(var(--gray-100) / <alpha-value>)",
          200: "rgb(var(--gray-200) / <alpha-value>)",
          300: "rgb(var(--gray-300) / <alpha-value>)",
          400: "rgb(var(--gray-400) / <alpha-value>)",
          500: "rgb(var(--gray-500) / <alpha-value>)",
          600: "rgb(var(--gray-600) / <alpha-value>)",
          700: "rgb(var(--gray-700) / <alpha-value>)",
          800: "rgb(var(--gray-800) / <alpha-value>)",
          900: "rgb(var(--gray-900) / <alpha-value>)",
        },
      },
      fontFamily: {
        sans: ['var(--font-et-book)', 'Charter', 'Bitstream Charter', 'Sitka Text', 'Cambria', 'serif'],
        serif: ['var(--font-et-book)', 'Charter', 'Bitstream Charter', 'Sitka Text', 'Cambria', 'serif'],
      },
    },
  },
  plugins: [],
  future: {
    hoverOnlyWhenSupported: true,
  },
};

export default config;
