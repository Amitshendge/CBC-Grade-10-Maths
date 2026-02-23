# Build stage
FROM node:20-alpine AS build

WORKDIR /app

# Copy package files
COPY code_base/ui/package*.json ./

# Install all dependencies (including dev dependencies needed for building)
RUN npm ci

# Copy source code
COPY code_base/ui/ ./

# Build the app
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy nginx configuration
COPY code_base/ui/nginx.conf /etc/nginx/conf.d/default.conf

# Copy built app from build stage (Create React App outputs to 'build' directory)
COPY --from=build /app/dist /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
