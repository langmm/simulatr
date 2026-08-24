import os
from beam import Image, Pod
_util_dir = os.path.abspath(os.path.dirname(__file__))


image = Image().from_dockerfile(
    os.path.join(_util_dir, "Dockerfile.server"),
    context_dir=os.path.dirname(_util_dir),
)

pod = Pod(
    name='apsimx-model',
    image=image,
    cpu=2,
    ports=[8000],
)

# res = pod.create()
# print(f"Pod at {res.url}")
