/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Local-dev only: when DEV_API_PROXY is set (and NEXT_PUBLIC_API_BASE is empty), the browser
  // calls this origin and Next forwards /api/* to the Flask server. Needed when the API runs
  // inside WSL and Windows does not forward its port. Inert in production builds.
  async rewrites() {
    const target = process.env.DEV_API_PROXY
    return target ? [{ source: '/api/:path*', destination: `${target}/api/:path*` }] : []
  },
}

export default nextConfig
