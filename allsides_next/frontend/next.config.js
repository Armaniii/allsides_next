/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  reactStrictMode: true,
  swcMinify: true,
  distDir: '.next',
  experimental: {
    turbo: {
      loaders: {
        '.js': ['swc-loader'],
        '.ts': ['swc-loader'],
        '.tsx': ['swc-loader'],
        '.jsx': ['swc-loader'],
      },
    },
  },
  poweredByHeader: false,
  // Specify the source directory
  // dir: 'src',
  async rewrites() {
    // Remove /api from the end if it exists
    const baseUrl = process.env.NEXT_PUBLIC_API_URL 
      ? process.env.NEXT_PUBLIC_API_URL.replace(/\/api$/, '')
      : 'http://localhost:9000';
    
    console.log('🔄 API Base URL for rewrites:', baseUrl);
    
    return [
      {
        source: '/api/token',
        destination: `${baseUrl}/api/token/`,
        has: [
          {
            type: 'header',
            key: 'content-type',
            value: 'application/json',
          },
        ],
      },
      {
        source: '/api/token/refresh',
        destination: `${baseUrl}/api/token/refresh/`,
        has: [
          {
            type: 'header',
            key: 'content-type',
            value: 'application/json',
          },
        ],
      },
      {
        source: '/api/:path*',
        destination: `${baseUrl}/api/:path*`,
      }
    ];
  },
  images: {
    domains: ['localhost', 'backend'],
  },
  // Server configuration
  serverRuntimeConfig: {
    port: parseInt(process.env.PORT, 10) || 3000,
  },
  publicRuntimeConfig: {
    staticFolder: '/static',
  },
  // Development optimization
  webpack: (config, { dev, isServer }) => {
    if (dev) {
      // Enable hot reload in development
      config.watchOptions = {
        poll: 1000, // Check for changes every second
        aggregateTimeout: 300, // Delay rebuild for 300ms
        ignored: [
          '**/.git/**',
          '**/node_modules/**',
          '**/.next/**',
          '**/dist/**',
          '**/build/**'
        ],
      };
    }
    return config;
  },
  // Enable hot reload for all environments
  webpackDevMiddleware: config => {
    config.watchOptions = {
      poll: 1000,
      aggregateTimeout: 300,
    }
    return config
  },
  // CORS configuration
  async headers() {
    return [
      {
        source: '/api/:path*',
        headers: [
          { key: 'Access-Control-Allow-Credentials', value: 'true' },
          { key: 'Access-Control-Allow-Origin', value: '*' },
          { key: 'Access-Control-Allow-Methods', value: 'GET,OPTIONS,PATCH,DELETE,POST,PUT' },
          { key: 'Access-Control-Allow-Headers', value: 'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version, Authorization' },
        ],
      }
    ];
  }
}

module.exports = nextConfig