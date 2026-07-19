"""
generate_mock_data.py
======================
Generates synthetic learning interaction data for 100+ mock users
and saves them directly as Gold Parquet files.

This solves the data sparsity and cold start issues for local testing and
model training in Jupyter notebooks.

Usage:
    python scripts/generate_mock_data.py --output-dir data/lakehouse/gold
"""

import os
import argparse
import numpy as np
import pandas as pd


def generate_data(num_users=100, num_courses=3, num_nodes=20):
    np.random.seed(42)
    
    user_ids = np.arange(1, num_users + 1)
    course_ids = np.arange(1, num_courses + 1)
    node_ids = np.arange(1001, 1001 + num_nodes)
    
    # 1. User-Item Affinity Matrix
    matrix_rows = []
    for user_id in user_ids:
        # Each user enrolls in 1 to 2 courses
        enrolled_courses = np.random.choice(course_ids, size=np.random.randint(1, 3), replace=False)
        for course_id in enrolled_courses:
            # Each user interacts with 3 to 12 nodes in the course
            interacted_nodes = np.random.choice(node_ids, size=np.random.randint(3, 13), replace=False)
            for node_id in interacted_nodes:
                total_interactions = np.random.randint(1, 15)
                # Weighted affinity score simulation
                implicit_affinity_score = float(total_interactions * np.random.uniform(0.8, 1.8))
                last_interaction_at = pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 10))
                
                matrix_rows.append({
                    "user_id": int(user_id),
                    "course_id": int(course_id),
                    "node_id": int(node_id),
                    "total_interactions": int(total_interactions),
                    "implicit_affinity_score": round(implicit_affinity_score, 2),
                    "last_interaction_at": last_interaction_at
                })
                
    df_matrix = pd.DataFrame(matrix_rows)
    
    # 2. Concept Struggles
    struggle_rows = []
    for _, row in df_matrix.sample(frac=0.25).iterrows():
        incorrect = np.random.randint(1, 5)
        correct = np.random.randint(0, incorrect) # incorrect > correct
        struggle_rate = float(incorrect / (correct + incorrect))
        
        struggle_rows.append({
            "user_id": int(row["user_id"]),
            "course_id": int(row["course_id"]),
            "node_id": int(row["node_id"]),
            "incorrect_checks_count": int(incorrect),
            "correct_checks_count": int(correct),
            "struggle_rate": round(struggle_rate, 2),
            "last_attempt_at": row["last_interaction_at"]
        })
        
    df_struggles = pd.DataFrame(struggle_rows)
    
    # 3. Student Course Metrics
    student_rows = []
    user_course_groups = df_matrix.groupby(["user_id", "course_id"])
    for (user_id, course_id), group in user_course_groups:
        viewed = len(group)
        completed = np.random.randint(0, viewed + 1)
        
        # quick checks
        user_struggles = df_struggles[
            (df_struggles["user_id"] == user_id) & (df_struggles["course_id"] == course_id)
        ]
        incorrect_checks = int(user_struggles["incorrect_checks_count"].sum())
        correct_checks = int(user_struggles["correct_checks_count"].sum())
        
        # Add some random correct checks for users who aren't struggling
        if len(user_struggles) == 0:
            correct_checks = np.random.randint(0, 8)
            incorrect_checks = 0
            
        total_checks = correct_checks + incorrect_checks
        check_accuracy = float(correct_checks / total_checks) if total_checks > 0 else 0.0
        
        ask_ai = np.random.randint(0, 5)
        flashcard_flips = np.random.randint(0, 20)
        
        # Learning style and engagement level simulation
        style = "Thực hành (Trắc nghiệm)" if correct_checks > viewed else "Đọc hiểu & Lý thuyết"
        if ask_ai > 2:
            style = "AI Mentor & Trao đổi"
            
        engagement = "Cần cố gắng"
        if len(group) > 8:
            engagement = "Rất tích cực"
        elif len(group) > 4:
            engagement = "Tích cực"
            
        rec_msg = "Nên làm thêm Quick Check để tự đánh giá."
        if check_accuracy < 0.6 and total_checks > 0:
            rec_msg = "Độ chính xác thấp, nên xem kỹ lại lý thuyết và thảo luận với AI."
            
        student_rows.append({
            "user_id": int(user_id),
            "course_id": int(course_id),
            "completed_lessons_count": int(completed),
            "viewed_lessons_count": int(viewed),
            "correct_checks_count": int(correct_checks),
            "incorrect_checks_count": int(incorrect_checks),
            "ask_ai_count": int(ask_ai),
            "flashcard_flips_count": int(flashcard_flips),
            "total_interactions_count": int(group["total_interactions"].sum()),
            "last_active_at": group["last_interaction_at"].max(),
            "check_accuracy": round(check_accuracy, 2),
            "learning_style": style,
            "engagement_level": engagement,
            "study_recommendation": rec_msg
        })
        
    df_student_metrics = pd.DataFrame(student_rows)
    
    # 4. Struggle Alerts
    alert_rows = []
    # 1. Concept struggle alerts
    for _, row in df_struggles.iterrows():
        alert_rows.append({
            "user_id": int(row["user_id"]),
            "course_id": int(row["course_id"]),
            "node_id": int(row["node_id"]),
            "alert_type": "concept_struggle",
            "alert_message": f"Học viên đang gặp khó khăn ở khái niệm (Khái niệm ID: {row['node_id']}) với tỷ lệ làm sai là {int(row['struggle_rate']*100)}%. Hãy ôn tập lại bài học!",
            "detected_at": row["last_attempt_at"]
        })
    # 2. Inactivity alerts
    for _, row in df_student_metrics[df_student_metrics["last_active_at"] < pd.Timestamp.now() - pd.Timedelta(days=7)].iterrows():
        alert_rows.append({
            "user_id": int(row["user_id"]),
            "course_id": int(row["course_id"]),
            "node_id": None,
            "alert_type": "inactivity",
            "alert_message": f"Đã lâu bạn chưa tham gia học tập trong khóa học (Course ID: {row['course_id']}). Hãy quay lại ôn luyện ngay nhé!",
            "detected_at": row["last_active_at"]
        })
        
    df_alerts = pd.DataFrame(alert_rows)
    
    # 5. Study Recommendations (Heuristics next best action)
    rec_rows = []
    for _, row in df_student_metrics.iterrows():
        user_id = row["user_id"]
        course_id = row["course_id"]
        
        # Check if they have a struggle concept
        user_struggles = df_struggles[
            (df_struggles["user_id"] == user_id) & (df_struggles["course_id"] == course_id)
        ]
        
        if len(user_struggles) > 0:
            weakest_node = int(user_struggles.loc[user_struggles["struggle_rate"].idxmax()]["node_id"])
            action = "review_struggle_concept"
            msg = f"Bạn đang gặp khó khăn ở khái niệm (ID: {weakest_node}). Hãy ôn tập lại lý thuyết bài học này!"
            node_id = weakest_node
        elif row["check_accuracy"] < 0.6 and (row["correct_checks_count"] + row["incorrect_checks_count"]) > 0:
            action = "discuss_with_ai"
            msg = "Cảnh báo: Độ chính xác Quick Check của bạn đang dưới 60%. Hãy thảo luận với AI Mentor để củng cố kiến thức."
            node_id = None
        else:
            action = "learn_next_lesson"
            msg = "Tiến độ học tập rất tốt! Hãy tiếp tục học bài học tiếp theo trong giáo trình."
            node_id = None
            
        rec_rows.append({
            "user_id": int(user_id),
            "course_id": int(course_id),
            "recommended_action_type": action,
            "recommended_node_id": node_id,
            "recommendation_message": msg,
            "generated_at": pd.Timestamp.now()
        })
        
    df_recs = pd.DataFrame(rec_rows)
    
    return df_matrix, df_struggles, df_student_metrics, df_alerts, df_recs


def main(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    print("[Generator] Generating synthetic data for 100 users...")
    df_matrix, df_struggles, df_student_metrics, df_alerts, df_recs = generate_data()
    
    # Save as Parquet files
    df_matrix.to_parquet(os.path.join(output_dir, "gold_user_item_matrix.parquet"), index=False)
    df_struggles.to_parquet(os.path.join(output_dir, "gold_concept_struggles.parquet"), index=False)
    df_student_metrics.to_parquet(os.path.join(output_dir, "gold_student_course_metrics.parquet"), index=False)
    df_alerts.to_parquet(os.path.join(output_dir, "gold_struggle_alerts.parquet"), index=False)
    df_recs.to_parquet(os.path.join(output_dir, "gold_study_recommendations.parquet"), index=False)
    
    print(f"[Generator] Gold Parquet files successfully saved to: {output_dir}")
    print(f"  gold_user_item_matrix.parquet: {len(df_matrix)} rows")
    print(f"  gold_concept_struggles.parquet: {len(df_struggles)} rows")
    print(f"  gold_student_course_metrics.parquet: {len(df_student_metrics)} rows")
    print(f"  gold_struggle_alerts.parquet: {len(df_alerts)} rows")
    print(f"  gold_study_recommendations.parquet: {len(df_recs)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic gold layer datasets for local testing")
    parser.add_argument("--output-dir", default="data/lakehouse/gold", help="Target output directory")
    args = parser.parse_args()
    main(args.output_dir)
