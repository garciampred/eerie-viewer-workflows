import os

import s3fs


def get_filesystem():
    fs = s3fs.S3FileSystem(
        key=os.environ["S3_KEY"],
        secret=os.environ["S3_SECRET"],
        client_kwargs=dict(endpoint_url=os.environ["S3_ENDPOINT_URL"]),
    )
    return fs
