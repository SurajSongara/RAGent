FROM node:22-slim

WORKDIR /app

# Dependencies first so a source edit does not reinstall the tree.
COPY package.json ./
RUN npm install

COPY . .

EXPOSE 3000
CMD ["npm", "run", "dev"]
