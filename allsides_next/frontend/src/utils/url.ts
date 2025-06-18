/**
 * Formats a URL string into a clean, readable domain name.
 * @param urlString The full URL to format.
 * @returns A clean domain name (e.g., "example.com") or the original string if parsing fails.
 */
export const formatUrlForDisplay = (urlString: string): string => {
  if (!urlString) return '';
  try {
    // The URL constructor requires a protocol. Prepend if missing.
    const fullUrl = /^(https?|ftp):\/\//.test(urlString) ? urlString : `https://${urlString}`;
    const url = new URL(fullUrl);
    return url.hostname.replace(/^www\./, '');
  } catch (error) {
    console.warn(`Could not parse URL for display: ${urlString}`);
    // Fallback to a simpler cleanup if URL constructor fails
    return urlString.replace(/^(https?:\/\/)?(www\.)?/, '').split('/')[0];
  }
};

/**
 * Normalizes a domain for comparison purposes.
 * @param input The domain string to normalize
 * @returns A normalized lowercase domain without www prefix
 */
export const normalizeDomain = (input: string): string => {
  try {
    // Use the URL constructor to reliably get the hostname
    const hostname = new URL(`https://${input.split('/')[0]}`).hostname;
    return hostname.replace(/^www\./, '').toLowerCase();
  } catch {
    // Fallback for simple strings that aren't valid host parts
    return input.replace(/^www\./, '').toLowerCase();
  }
};