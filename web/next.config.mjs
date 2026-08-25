/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // pdf.js ships Node-only fallbacks that must not be bundled for the browser.
  webpack: (config) => {
    config.resolve.alias.canvas = false;
    return config;
  },
};

export default nextConfig;
