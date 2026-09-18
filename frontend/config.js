/**
 * config.js - Dynamic API Resolver for Vercel Frontend & Render Backend
 */
(function() {
    const DEFAULT_PROD_RENDER_URL = "https://full-stack-multi-noise-classifcation.onrender.com";
    const DEFAULT_LOCAL_URL = "http://localhost:5000";

    // Auto-detect environment
    function getStoredBackendUrl() {
        const stored = localStorage.getItem('LUNGCT_API_BASE_URL');
        if (stored && stored.trim() !== '') {
            return stored.trim().replace(/\/+$/, '');
        }
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            return DEFAULT_LOCAL_URL;
        }
        return DEFAULT_PROD_RENDER_URL;
    }

    window.API_CONFIG = {
        getBaseUrl: getStoredBackendUrl,
        setBaseUrl: function(url) {
            if (url) {
                const cleanUrl = url.trim().replace(/\/+$/, '');
                localStorage.setItem('LUNGCT_API_BASE_URL', cleanUrl);
            } else {
                localStorage.removeItem('LUNGCT_API_BASE_URL');
            }
        },
        getUrl: function(path) {
            const base = getStoredBackendUrl();
            const cleanPath = path.startsWith('/') ? path : '/' + path;
            return `${base}${cleanPath}`;
        }
    };
})();
