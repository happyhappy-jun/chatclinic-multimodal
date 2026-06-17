import type { NextConfig } from "next";

// Proxy /api/* to the backend so the browser only ever talks to the frontend
// origin. This keeps the backend (:8001) and model server (:8000) private and
// makes a single-port tunnel (ngrok http 3000) work with no CORS/host juggling.
const BACKEND_ORIGIN = process.env.BACKEND_ORIGIN || "http://127.0.0.1:8001";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${BACKEND_ORIGIN}/api/:path*` }];
  },
};

export default nextConfig;
