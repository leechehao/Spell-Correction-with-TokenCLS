# json
curl http://127.0.0.1:9488/invocations -X POST -H "Content-Type: application/json" -d '{
    "dataframe_split": {
        "columns": ["text"],
        "data": ["nodul in righ upper lung .", "mass in left upper lung .", "opciy and noodl at lung ."]
    }
}'

# csv
curl http://127.0.0.1:9488/invocations -X POST -H "Content-Type: text/csv" --data-binary "@post_file.csv"
