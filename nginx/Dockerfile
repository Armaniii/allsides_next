FROM nginx:1.25-alpine

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom configuration
COPY conf.d/default.conf /etc/nginx/conf.d/

EXPOSE 80 443 