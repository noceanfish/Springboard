# ==================================== BASE ====================================
# Use the official image as a parent image.
FROM python:3.6.12-slim AS base

# update pip
RUN apt-get update -y && \
    apt-get install -y python-pip python-dev

# Set the working directory.
WORKDIR /app

# Copy the file from your host to your current location.
COPY requirements.txt /app/requirements.txt

# add user
#RUN useradd -m sid
#RUN chown -R sid:sid /app
#USER sid
#ENV PATH="/home/sid/.local/bin:${PATH}"

# ==================================== Production ====================================
FROM base AS production

# Run the command inside your image filesystem.
#RUN pip --no-cache-dir install --user -r requirements.txt
RUN pip --no-cache-dir install -r requirements.txt
RUN python3 -m spacy download en_core_web_md
RUN python3 -m nltk.downloader stopwords

COPY . /app

# Add metadata to the image to describe which port the container is listening on at runtime.
EXPOSE 5000

# Setup running environment, and
# Run the specified command within the container.
ENV FLASK_APP=stk_predictor.app
ENTRYPOINT [ "flask" ]
CMD [ "run", "--host=0.0.0.0" ]