# Google Cloud Run Deployment

This setup keeps the repository public while storing private match data in a private Google Cloud Storage bucket.

## 1. Create a private bucket

```bash
gcloud storage buckets create gs://YOUR_BUCKET_NAME --location=europe-west1 --uniform-bucket-level-access
```

Upload private runtime files using the same folder names the app expects:

```bash
gcloud storage cp --recursive raw_data gs://YOUR_BUCKET_NAME/raw_data
gcloud storage cp --recursive assets gs://YOUR_BUCKET_NAME/assets
gcloud storage cp --recursive göztepehub/assets gs://YOUR_BUCKET_NAME/göztepehub/assets
gcloud storage cp utils/tactiq_xg_model.json gs://YOUR_BUCKET_NAME/utils/tactiq_xg_model.json
```

Do not make the bucket public.

## 2. Deploy the main tactIQ app

```bash
gcloud run deploy tactiq \
  --source . \
  --region europe-west1 \
  --allow-unauthenticated \
  --set-env-vars TACTIQ_DATA_BUCKET=YOUR_BUCKET_NAME
```

## 3. Deploy the Goztepe Hub app

```bash
gcloud run deploy tactiq-goztepehub \
  --source . \
  --region europe-west1 \
  --allow-unauthenticated \
  --set-env-vars TACTIQ_DATA_BUCKET=YOUR_BUCKET_NAME,APP_MODULE=goztepehub_app:server
```

## 4. Connect the navbar links

After both services are deployed, copy their Cloud Run URLs and update the environment variables:

```bash
gcloud run services update tactiq \
  --region europe-west1 \
  --set-env-vars GOZTEPE_HUB_URL=https://YOUR-GOZTEPE-HUB-URL

gcloud run services update tactiq-goztepehub \
  --region europe-west1 \
  --set-env-vars TACTIQ_MAIN_URL=https://YOUR-TACTIQ-URL
```

## Notes

- The raw files stay in the private bucket. Users only see the Dash pages.
- Keep Cloud Run minimum instances at `0` to avoid idle cost.
- Add a billing budget alert before sharing the public link.
