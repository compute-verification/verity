# The image node 1's Kueue jobs run in (dispatch.py's, n1_lease.py's and n1_custody.sh's IMAGE): SkyPilot's May GPU image with
# its trust store upgraded, since that Ubuntu 22.04 base still trusts the roots that cross-signed the compromised eMudhra
# intermediates (USN-8944-1; INF-314), and the jobs' Python reads the container's /etc/ssl/certs. The pods set their own user.
#   IMAGE_NAME=verity-sky-gpu DOCKERFILE=sky-gpu.Dockerfile build.sh
FROM us-docker.pkg.dev/sky-dev-465/skypilotk8s/skypilot-gpu@sha256:3c45c3a1eac25ba8753d7f940ccb99c8d648fd315da368adabe04179ad918890

USER root
RUN apt-get update -q \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y -q --only-upgrade ca-certificates 'libnss3*' \
 && rm -rf /var/lib/apt/lists/* \
 && ! ls /etc/ssl/certs | grep -E 'emSign_ECC_Root_CA_-_G3|Comodo_AAA_Services_root'
