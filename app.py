import streamlit as st
import pandas as pd
import random
import copy
import numpy as np
from sklearn.cluster import KMeans

# --- SETTINGS ---
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TIMES = ["09:00", "10:00", "11:00", "12:00", "14:00"]
POPULATION_SIZE = 100
GENERATIONS = 200

class TimetableAI:
    def __init__(self, teacher_subject_map, rooms):
        self.teacher_subject_map = teacher_subject_map
        self.teachers = list(teacher_subject_map.keys())
        self.teacher_options = self.teachers + ["NONE"] 
        self.rooms = rooms

    def generate_random_schedule(self):
        schedule = []
        for day in DAYS:
            for time in TIMES:
                for room in self.rooms:
                    teacher = random.choice(self.teacher_options)
                    subject = self.teacher_subject_map.get(teacher, "---")
                    schedule.append({"Day": day, "Time": time, "Room": room, "Subject": subject, "Teacher": teacher})
        return schedule

    def calculate_fitness(self, schedule):
        conflicts = 0
        time_slot_usage = {} 
        teacher_daily_stats = {} 

        for slot in schedule:
            if slot['Teacher'] == "NONE": continue
            day_time = (slot['Day'], slot['Time'])
            teacher = slot['Teacher']
            if day_time not in time_slot_usage: time_slot_usage[day_time] = []
            if teacher in time_slot_usage[day_time]:
                conflicts += 500  
            time_slot_usage[day_time].append(teacher)
            t_day_key = (teacher, slot['Day'])
            if t_day_key not in teacher_daily_stats: teacher_daily_stats[t_day_key] = []
            teacher_daily_stats[t_day_key].append(slot['Time'])

        for t_day, assigned_times in teacher_daily_stats.items():
            if len(assigned_times) > 3: conflicts += 100
            if len(assigned_times) > 1:
                indices = sorted([TIMES.index(t) for t in assigned_times])
                for i in range(len(indices) - 1):
                    if indices[i+1] - indices[i] == 1:
                        conflicts += 100 
        return 1 / (1 + conflicts)

    def evolve(self):
        population = [self.generate_random_schedule() for _ in range(POPULATION_SIZE)]
        for i in range(GENERATIONS):
            population = sorted(population, key=lambda x: self.calculate_fitness(x), reverse=True)
            if self.calculate_fitness(population[0]) == 1.0: break
            next_gen = population[:15]
            while len(next_gen) < POPULATION_SIZE:
                parent = copy.deepcopy(random.choice(population[:20]))
                for _ in range(3):
                    idx = random.randint(0, len(parent)-1)
                    new_t = random.choice(self.teacher_options)
                    parent[idx]['Teacher'] = new_t
                    parent[idx]['Subject'] = self.teacher_subject_map.get(new_t, "---")
                next_gen.append(parent)
            population = next_gen
        return population[0]

# --- UI ---
st.set_page_config(page_title="AI Timetable & ML Analytics", layout="wide")
st.title("📅 AI-Powered Time Table Generator with Smart Recommendations")

with st.sidebar:
    st.header("1. Input Parameters")
    raw_input = st.text_area("Teacher: Subject List", "Dr. Smith: Math\nProf. Jones: Physics\nMs. Davis: CS\nMr. Wilson: History\nDr. Brown: Biology")
    teacher_subject_map = {line.split(":")[0].strip(): line.split(":")[1].strip() for line in raw_input.split('\n') if ":" in line}
    rooms = st.text_area("Available Rooms", "Room A, Room B, Room C").split(",")
    rooms = [r.strip() for r in rooms if r.strip()]
    generate_btn = st.button("Generate Optimized Timetable", type="primary")

if generate_btn:
    engine = TimetableAI(teacher_subject_map, rooms)
    best_schedule = engine.evolve()
    st.session_state['master_schedule'] = best_schedule

if 'master_schedule' in st.session_state:
    df = pd.DataFrame(st.session_state['master_schedule'])
    
    # --- MASTER VIEW ---
    st.header("🏫 Master Institution Schedule")
    df["Display"] = df.apply(lambda x: "--- EMPTY ---" if x["Teacher"] == "NONE" else f"{x['Subject']} ({x['Teacher']})", axis=1)
    master_pivot = df.pivot_table(index=['Day', 'Time'], columns='Room', values='Display', aggfunc='first').reindex(DAYS, level=0)
    st.table(master_pivot)

    # --- K-MEANS CLUSTERING SECTION ---
    st.divider()
    st.header("🤖 ML Workload Analysis & Recommendations")
    
    workload_data = df[df['Teacher'] != 'NONE']['Teacher'].value_counts().reindex(teacher_subject_map.keys(), fill_value=0)
    X = workload_data.values.reshape(-1, 1)

    # Clustering
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(X)

    # Naming Clusters Logically
    cluster_centers = kmeans.cluster_centers_.flatten()
    if cluster_centers[0] > cluster_centers[1]:
        names = {0: "🔥 Heavy Workload", 1: "✅ Optimal Workload"}
    else:
        names = {1: "🔥 Heavy Workload", 0: "✅ Optimal Workload"}

    analysis_df = pd.DataFrame({
        "Teacher": workload_data.index,
        "Total Classes": workload_data.values,
        "Status": [names[c] for c in clusters]
    })

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Workload Distribution")
        st.bar_chart(analysis_df.set_index("Teacher")["Total Classes"])
        st.dataframe(analysis_df, use_container_width=True)

    with col2:
        st.subheader("💡 AI Recommendations")
        heavy_teachers = analysis_df[analysis_df["Status"] == "🔥 Heavy Workload"]["Teacher"].tolist()
        optimal_teachers = analysis_df[analysis_df["Status"] == "✅ Optimal Workload"]["Teacher"].tolist()
        
        if heavy_teachers:
            st.warning(f"**Action Required:** {', '.join(heavy_teachers)} have been identified in the Heavy Workload cluster.")
            st.write("- Consider assigning a Teaching Assistant (TA) to these faculty members.")
            st.write("- Check if any elective subjects can be moved to a different semester.")
        
        if len(optimal_teachers) > len(heavy_teachers):
            st.info("**Resource Suggestion:** You have spare capacity in the Optimal Workload group. You can safely assign substitute duties to these teachers if needed.")

    # --- TEACHER PORTALS ---
    st.divider()
    st.header("👨‍🏫 Teacher Portals")
    selected_t = st.selectbox("View Timetable For:", list(teacher_subject_map.keys()))
    t_df = df[df['Teacher'] == selected_t].copy()
    t_df["Room_Assign"] = t_df["Subject"] + " [" + t_df["Room"] + "]"
    
    if not t_df.empty:
        t_pivot = t_df.pivot_table(index='Time', columns='Day', values='Room_Assign', aggfunc='first').reindex(index=TIMES, columns=DAYS)
        st.table(t_pivot.fillna("--- REST / BREAK ---"))