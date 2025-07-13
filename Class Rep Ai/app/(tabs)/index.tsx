import React, { useEffect, useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ScrollView } from 'react-native';
import { router } from 'expo-router';
import { useData } from '@/context/DataContext';
import { useTheme } from '@/context/ThemeContext';
import { Users, BookOpen, CircleCheck as CheckCircle, CircleAlert as AlertCircle, Plus, ChartBar as BarChart3 } from 'lucide-react-native';

export default function HomeScreen() {
  const { getUserData, getTodayAttendance, getUpcomingAssignments } = useData();
  const { theme } = useTheme();
  const [userData, setUserData] = useState<any>(null);
  const [todayStats, setTodayStats] = useState({ present: 0, absent: 0, total: 0 });
  const [upcomingAssignments, setUpcomingAssignments] = useState([]);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    const user = await getUserData();
    setUserData(user);
    
    const attendance = await getTodayAttendance();
    const stats = attendance.reduce((acc: any, record: any) => {
      acc.total++;
      if (record.status === 'present') acc.present++;
      if (record.status === 'absent') acc.absent++;
      return acc;
    }, { present: 0, absent: 0, total: 0 });
    setTodayStats(stats);

    const assignments = await getUpcomingAssignments();
    setUpcomingAssignments(assignments.slice(0, 3));
  };

  const styles = createStyles(theme);

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.greeting}>Welcome back,</Text>
        <Text style={styles.userName}>{userData?.studentName}</Text>
        <Text style={styles.rollNumber}>Roll No: {userData?.rollNumber}</Text>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>📋 Today's Attendance</Text>
        <View style={styles.card}>
          <View style={styles.statsRow}>
            <View style={styles.statItem}>
              <CheckCircle size={24} color="#16A34A" />
              <Text style={styles.statNumber}>{todayStats.present}</Text>
              <Text style={styles.statLabel}>Present</Text>
            </View>
            <View style={styles.statItem}>
              <AlertCircle size={24} color="#DC2626" />
              <Text style={styles.statNumber}>{todayStats.absent}</Text>
              <Text style={styles.statLabel}>Absent</Text>
            </View>
            <View style={styles.statItem}>
              <Users size={24} color="#2563EB" />
              <Text style={styles.statNumber}>{todayStats.total}</Text>
              <Text style={styles.statLabel}>Total</Text>
            </View>
          </View>
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>📆 Upcoming Assignments</Text>
        <View style={styles.card}>
          {upcomingAssignments.length > 0 ? (
            upcomingAssignments.map((assignment: any, index) => (
              <View key={index} style={styles.assignmentItem}>
                <Text style={styles.assignmentSubject}>{assignment.subject}</Text>
                <Text style={styles.assignmentTitle}>{assignment.requirement}</Text>
                <Text style={styles.assignmentDue}>Due: {assignment.dueDate}</Text>
              </View>
            ))
          ) : (
            <Text style={styles.emptyText}>No upcoming assignments</Text>
          )}
        </View>
      </View>

      <View style={styles.actions}>
        <TouchableOpacity 
          style={[styles.actionButton, { backgroundColor: '#16A34A' }]}
          onPress={() => router.push('/(tabs)/attendance')}
        >
          <CheckCircle size={24} color="#FFFFFF" />
          <Text style={styles.actionButtonText}>Mark Attendance</Text>
        </TouchableOpacity>

        <TouchableOpacity 
          style={[styles.actionButton, { backgroundColor: '#2563EB' }]}
          onPress={() => router.push('/(tabs)/assignments')}
        >
          <Plus size={24} color="#FFFFFF" />
          <Text style={styles.actionButtonText}>Add Assignment</Text>
        </TouchableOpacity>

        <TouchableOpacity 
          style={[styles.actionButton, { backgroundColor: '#EA580C' }]}
          onPress={() => router.push('/(tabs)/reports')}
        >
          <BarChart3 size={24} color="#FFFFFF" />
          <Text style={styles.actionButtonText}>View Reports</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
}

const createStyles = (theme: any) => StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.background,
  },
  header: {
    padding: 24,
    paddingTop: 60,
  },
  greeting: {
    fontSize: 18,
    color: theme.textSecondary,
  },
  userName: {
    fontSize: 28,
    fontWeight: 'bold',
    color: theme.text,
    marginTop: 4,
  },
  rollNumber: {
    fontSize: 16,
    color: theme.textSecondary,
    marginTop: 4,
  },
  section: {
    padding: 24,
    paddingTop: 0,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: theme.text,
    marginBottom: 12,
  },
  card: {
    backgroundColor: theme.surface,
    borderRadius: 16,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 3,
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  statItem: {
    alignItems: 'center',
  },
  statNumber: {
    fontSize: 24,
    fontWeight: 'bold',
    color: theme.text,
    marginTop: 8,
  },
  statLabel: {
    fontSize: 14,
    color: theme.textSecondary,
    marginTop: 4,
  },
  assignmentItem: {
    borderBottomWidth: 1,
    borderBottomColor: theme.border,
    paddingBottom: 12,
    marginBottom: 12,
  },
  assignmentSubject: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2563EB',
  },
  assignmentTitle: {
    fontSize: 14,
    color: theme.text,
    marginTop: 4,
  },
  assignmentDue: {
    fontSize: 12,
    color: theme.textSecondary,
    marginTop: 4,
  },
  emptyText: {
    fontSize: 16,
    color: theme.textSecondary,
    textAlign: 'center',
    fontStyle: 'italic',
  },
  actions: {
    padding: 24,
    paddingTop: 0,
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    borderRadius: 12,
    marginBottom: 12,
  },
  actionButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: 'bold',
    marginLeft: 8,
  },
});