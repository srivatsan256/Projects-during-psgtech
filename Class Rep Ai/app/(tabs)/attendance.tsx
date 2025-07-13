import React, { useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ScrollView, TextInput, Modal } from 'react-native';
import { useData } from '@/context/DataContext';
import { useTheme } from '@/context/ThemeContext';
import { CircleCheck as CheckCircle, Circle as XCircle, TriangleAlert as AlertTriangle, Download } from 'lucide-react-native';

const VALID_ROLL_NUMBERS = [
  ...Array.from({ length: 69 }, (_, i) => 201 + i).filter(n => n !== 248),
  431, 432, 433, 434, 435, 436
];

export default function AttendanceScreen() {
  const { getStudents, saveAttendance, getAttendanceByDate, exportAttendance } = useData();
  const { theme } = useTheme();
  const [students, setStudents] = useState<any[]>([]);
  const [attendance, setAttendance] = useState<any>({});
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);
  const [showExceptionModal, setShowExceptionModal] = useState(false);
  const [selectedStudent, setSelectedStudent] = useState<any>(null);
  const [exceptionReason, setExceptionReason] = useState('');

  useEffect(() => {
    loadStudents();
    loadAttendance();
  }, [selectedDate]);

  const loadStudents = async () => {
    let studentList = await getStudents();
    if (studentList.length === 0) {
      // Initialize with valid roll numbers
      studentList = VALID_ROLL_NUMBERS.map(rollNo => ({
        rollNumber: `22N${rollNo}`,
      }));
      // Save to storage for future use
      for (const student of studentList) {
        await saveStudent(student);
      }
    }
    setStudents(studentList);
  };

  const loadAttendance = async () => {
    const attendanceData = await getAttendanceByDate(selectedDate);
    const attendanceMap = attendanceData.reduce((acc: any, record: any) => {
      acc[record.rollNumber] = record;
      return acc;
    }, {});
    setAttendance(attendanceMap);
  };

  const saveStudent = async (student: any) => {
    // This would save to your student database
    // Implementation depends on your data storage
  };

  const markAttendance = async (student: any, status: string, reason?: string) => {
    const record = {
      rollNumber: student.rollNumber,
      date: selectedDate,
      status,
      reason: reason || '',
    };

    await saveAttendance(record);
    setAttendance(prev => ({
      ...prev,
      [student.rollNumber]: record,
    }));
  };

  const handleStatusPress = (student: any, status: string) => {
    if (status === 'exception') {
      setSelectedStudent(student);
      setShowExceptionModal(true);
    } else {
      markAttendance(student, status);
    }
  };

  const handleExceptionSave = () => {
    if (selectedStudent && exceptionReason.trim()) {
      markAttendance(selectedStudent, 'exception', exceptionReason);
      setShowExceptionModal(false);
      setExceptionReason('');
      setSelectedStudent(null);
    }
  };

  const handleExport = async () => {
    await exportAttendance(selectedDate);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'present': return '#16A34A';
      case 'absent': return '#DC2626';
      case 'exception': return '#EA580C';
      default: return theme.border;
    }
  };

  const styles = createStyles(theme);

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Attendance Manager</Text>
        <TouchableOpacity style={styles.exportButton} onPress={handleExport}>
          <Download size={20} color="#FFFFFF" />
          <Text style={styles.exportButtonText}>Export</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.dateSection}>
        <TextInput
          style={styles.dateInput}
          value={selectedDate}
          onChangeText={setSelectedDate}
          placeholder="YYYY-MM-DD"
        />
      </View>

      <ScrollView style={styles.studentList}>
        {students.map((student) => {
          const record = attendance[student.rollNumber];
          const status = record?.status;

          return (
            <View key={student.rollNumber} style={styles.studentItem}>
              <View style={styles.studentInfo}>
                <Text style={styles.rollNumber}>{student.rollNumber}</Text>
                {status === 'exception' && record.reason && (
                  <Text style={styles.exceptionReason}>{record.reason}</Text>
                )}
              </View>

              <View style={styles.statusButtons}>
                <TouchableOpacity
                  style={[
                    styles.statusButton,
                    { backgroundColor: status === 'present' ? '#16A34A' : theme.surface }
                  ]}
                  onPress={() => handleStatusPress(student, 'present')}
                >
                  <CheckCircle size={20} color={status === 'present' ? '#FFFFFF' : '#16A34A'} />
                </TouchableOpacity>

                <TouchableOpacity
                  style={[
                    styles.statusButton,
                    { backgroundColor: status === 'absent' ? '#DC2626' : theme.surface }
                  ]}
                  onPress={() => handleStatusPress(student, 'absent')}
                >
                  <XCircle size={20} color={status === 'absent' ? '#FFFFFF' : '#DC2626'} />
                </TouchableOpacity>

                <TouchableOpacity
                  style={[
                    styles.statusButton,
                    { backgroundColor: status === 'exception' ? '#EA580C' : theme.surface }
                  ]}
                  onPress={() => handleStatusPress(student, 'exception')}
                >
                  <AlertTriangle size={20} color={status === 'exception' ? '#FFFFFF' : '#EA580C'} />
                </TouchableOpacity>
              </View>
            </View>
          );
        })}
      </ScrollView>

      <Modal visible={showExceptionModal} transparent animationType="slide">
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Exception Reason</Text>
            <TextInput
              style={styles.reasonInput}
              value={exceptionReason}
              onChangeText={setExceptionReason}
              placeholder="Enter reason for exception"
              multiline
            />
            <View style={styles.modalButtons}>
              <TouchableOpacity
                style={[styles.modalButton, styles.cancelButton]}
                onPress={() => setShowExceptionModal(false)}
              >
                <Text style={styles.cancelButtonText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.modalButton, styles.saveButton]}
                onPress={handleExceptionSave}
              >
                <Text style={styles.saveButtonText}>Save</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const createStyles = (theme: any) => StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.background,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 24,
    paddingTop: 60,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: theme.text,
  },
  exportButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#2563EB',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
  },
  exportButtonText: {
    color: '#FFFFFF',
    marginLeft: 4,
    fontWeight: '600',
  },
  dateSection: {
    paddingHorizontal: 24,
    paddingBottom: 16,
  },
  dateInput: {
    borderWidth: 1,
    borderColor: theme.border,
    borderRadius: 8,
    padding: 12,
    backgroundColor: theme.surface,
    color: theme.text,
  },
  studentList: {
    flex: 1,
    paddingHorizontal: 24,
  },
  studentItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: theme.surface,
    padding: 16,
    borderRadius: 12,
    marginBottom: 12,
  },
  studentInfo: {
    flex: 1,
  },
  rollNumber: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2563EB',
  },
  studentName: {
    fontSize: 12,
    color: '#EA580C',
    marginTop: 4,
    fontStyle: 'italic',
  },
  statusButtons: {
    flexDirection: 'row',
    gap: 8,
  },
  statusButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: theme.border,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    backgroundColor: theme.surface,
    padding: 24,
    borderRadius: 16,
    width: '90%',
    maxWidth: 400,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: theme.text,
    marginBottom: 16,
  },
  reasonInput: {
    borderWidth: 1,
    borderColor: theme.border,
    borderRadius: 8,
    padding: 12,
    minHeight: 80,
    textAlignVertical: 'top',
    color: theme.text,
    marginBottom: 16,
  },
  modalButtons: {
    flexDirection: 'row',
    gap: 12,
  },
  modalButton: {
    flex: 1,
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  cancelButton: {
    backgroundColor: theme.border,
  },
  saveButton: {
    backgroundColor: '#EA580C',
  },
  cancelButtonText: {
    color: theme.text,
    fontWeight: '600',
  },
  saveButtonText: {
    color: '#FFFFFF',
    fontWeight: '600',
  },
});