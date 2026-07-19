# This directory stores Gold Parquet exports and Bronze historical partitions.
# It is NOT committed to git.
#
# To populate this directory, either:
#   1. Create a symlink to the personalize-service shared volume:
#      ln -s /path/to/personalize-service/data/lakehouse ./data/lakehouse
#
#   2. Or trigger the export endpoint:
#      curl -X POST -H "X-AI-Secret: <secret>" http://localhost:8085/personalize/analytics/gold/export
#      Then copy the files from the server volume.
#
# Expected structure after population:
#   data/lakehouse/gold/gold_student_course_metrics.parquet
#   data/lakehouse/gold/gold_concept_struggles.parquet
#   data/lakehouse/gold/gold_user_item_matrix.parquet
#   data/lakehouse/gold/gold_struggle_alerts.parquet
#   data/lakehouse/gold/gold_study_recommendations.parquet
#   data/lakehouse/bronze/interactions/year=YYYY/month=MM/...
