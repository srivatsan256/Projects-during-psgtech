import React, { useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ScrollView, TextInput } from 'react-native';
import { useData } from '@/context/DataContext';
import { useTheme } from '@/context/ThemeContext';
import { ChartBar as BarChart3, Download, Share, Copy, Calendar, Users } from 'lucide-react-native';

export default function ReportsScreen() {
  const { getAttendanceReport, getAssignmentReport, exportAttendance, exportAssignments } = useData();
  const { theme } = useTheme();
  const [attendanceStats, setAttendanceStats] = useState<any>({});
  const [assignmentStats, setAssignmentStats] = useState<any>({});
  const [dateRange, setDateRange] = useState({
    start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    end: new Date().toISOString().split('T')[0],
  });

  useEffect(() => {
    loadReports();
  }, [dateRange]);

  const loadReports = async () => {
    const attendance = await getAttendanceReport(dateRange.start, dateRange.end);
    const assignments = await getAssignmentReport();
    
    setAttendanceStats(attendance);
    setAssignmentStats(assignments);
  };

  const handleExportAttendance = async (format: string) => {
    await exportAttendance(dateRange.start, dateRange.end, format);
  };

  const handleExportAssignments = async () => {
    await exportAssignments();
  };

  const styles = createStyles(theme);

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Reports & Analytics</Text>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>📅 Date Range</Text>
        <View style={styles.dateRange}>
          <TextInput
            style={styles.dateInput}
            value={dateRange.start}
            onChangeText={(text) => setDateRange(prev => ({ ...prev, start: text }))}
            placeholder="Start Date"
          />
          <Text style={styles.dateSeparator}>to</Text>
          <TextInput
            style={styles.dateInput}
            value={dateRange.end}
            onChangeText={(text) => setDateRange(prev => ({ ...prev, end: text }))}
            placeholder="End Date"
          />
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>📋 Attendance Summary</Text>
        <View style={styles.card}>
          <View style={styles.statRow}>
            <View style={styles.statItem}>
              <Text style={styles.statNumber}>{attendanceStats.totalDays || 0}</Text>
              <Text style={styles.statLabel}>Total Days</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statNumber}>{attendanceStats.avgPresent || 0}%</Text>
              <Text style={styles.statLabel}>Avg Attendance</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statNumber}>{attendanceStats.totalStudents || 0}</Text>
              <Text style={styles.statLabel}>Students</Text>
            </View>
          </View>

          <View style={styles.exportSection}>
            <Text style={styles.exportTitle}>Export Attendance:</Text>
            <View style={styles.exportButtons}>
              <TouchableOpacity
                style={[styles.exportButton, { backgroundColor: '#16A34A' }]}
                onPress={() => handleExportAttendance('whatsapp')}
              >
                <Share size={16} color="#FFFFFF" />
                <Text style={styles.exportButtonText}>WhatsApp</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.exportButton, { backgroundColor: '#DC2626' }]}
                onPress={() => handleExportAttendance('pdf')}
              >
                <Download size={16} color="#FFFFFF" />
                <Text style={styles.exportButtonText}>PDF</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.exportButton, { backgroundColor: '#2563EB' }]}
                onPress={() => handleExportAttendance('csv')}
              >
                <Download size={16} color="#FFFFFF" />
                <Text style={styles.exportButtonText}>CSV</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>📝 Assignment Summary</Text>
        <View style={styles.card}>
          <View style={styles.statRow}>
            <View style={styles.statItem}>
              <Text style={styles.statNumber}>{assignmentStats.total || 0}</Text>
              <Text style={styles.statLabel}>Total</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statNumber}>{assignmentStats.upcoming || 0}</Text>
              <Text style={styles.statLabel}>Upcoming</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statNumber}>{assignmentStats.overdue || 0}</Text>
              <Text style={styles.statLabel}>Overdue</Text>
            </View>
          </View>

          <View style={styles.exportSection}>
            <Text style={styles.exportTitle}>Export Assignments:</Text>
            <TouchableOpacity
              style={[styles.exportButton, { backgroundColor: '#16A34A', alignSelf: 'flex-start' }]}
              onPress={handleExportAssignments}
            >
              <Share size={16} color="#FFFFFF" />
              <Text style={styles.exportButtonText}>WhatsApp Format</Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>📊 Quick Actions</Text>
        <View style={styles.quickActions}>
          <TouchableOpacity style={styles.quickAction}>
            <Copy size={24} color="#2563EB" />
            <Text style={styles.quickActionText}>Copy All Data</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.quickAction}>
            <BarChart3 size={24} color="#16A34A" />
            <Text style={styles.quickActionText}>Generate Report</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.quickAction}>
            <Calendar size={24} color="#EA580C" />
            <Text style={styles.quickActionText}>Monthly View</Text>
          </TouchableOpacity>
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>📈 Recent Activity</Text>
        <View style={styles.card}>
          <Text style={styles.activityItem}>• Attendance marked for {new Date().toLocaleDateString()}</Text>
          <Text style={styles.activityItem}>• {assignmentStats.recent || 0} assignments added this week</Text>
          <Text style={styles.activityItem}>• Last export: {attendanceStats.lastExport || 'Never'}</Text>
        </View>
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
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: theme.text,
  },
  section: {
    padding: 24,
    paddingTop: 0,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: theme.text,
    marginBottom: 16,
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
  dateRange: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  dateInput: {
    flex: 1,
    borderWidth: 1,
    borderColor: theme.border,
    borderRadius: 8,
    padding: 12,
    backgroundColor: theme.surface,
    color: theme.text,
  },
  dateSeparator: {
    color: theme.textSecondary,
    fontWeight: '600',
  },
  statRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 20,
  },
  statItem: {
    alignItems: 'center',
  },
  statNumber: {
    fontSize: 24,
    fontWeight: 'bold',
    color: theme.text,
  },
  statLabel: {
    fontSize: 14,
    color: theme.textSecondary,
    marginTop: 4,
  },
  exportSection: {
    borderTopWidth: 1,
    borderTopColor: theme.border,
    paddingTop: 16,
  },
  exportTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: theme.text,
    marginBottom: 12,
  },
  exportButtons: {
    flexDirection: 'row',
    gap: 8,
    flexWrap: 'wrap',
  },
  exportButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
    gap: 4,
  },
  exportButtonText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '600',
  },
  quickActions: {
    flexDirection: 'row',
    gap: 12,
    flexWrap: 'wrap',
  },
  quickAction: {
    flex: 1,
    backgroundColor: theme.surface,
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    minWidth: 100,
  },
  quickActionText: {
    color: theme.text,
    fontSize: 12,
    fontWeight: '600',
    marginTop: 8,
    textAlign: 'center',
  },
  activityItem: {
    color: theme.textSecondary,
    fontSize: 14,
    marginBottom: 8,
    lineHeight: 20,
  },
});