# Minimal development image for quantumHams.
#
# The project source is not copied into the image — tools/run.sh mounts the
# project root into /workspace at runtime, so rebuilding the image is only
# needed when this Dockerfile itself changes. entrypoint.sh installs
# quantumhams in editable mode against the mounted source on every container
# start, so edits made on the host (under src/quantumhams) are picked up
# immediately without rebuilding the image.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["entrypoint.sh"]
CMD ["/bin/bash"]
