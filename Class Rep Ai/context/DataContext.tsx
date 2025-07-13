import React, { createContext, useContext } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as Sharing from 'expo-sharing';
import * as FileSystem from 'expo-file-system';

interface DataContextType {
  // User Management
  saveUserData: (userData: any) => Promise<void>;
  getUserData: () => Promise<any>;
  
  // Student Management
  getStudents: () => Promise<any[]>;
  
  // Attendance Management
  saveAttendance: (record: any) => Promise<void>;
  getAttendanceByDate: (date: string) => Promise<any[]>;
  getTodayAttendance: () => Promise<any[]>;
  getAttendanceReport: (startDate: string, endDate: string) => Promise<any>;
  exportAttendance: (startDate?: string, endDate?: string, format?: string) => Promise<void>;
  
  // Assignment Management
  getAssignments: () => Promise<any[]>;
  saveAssignment: (assignment: any) => Promise<void>;
  deleteAssignment: (id: string) => Promise<void>;
  getUpcomingAssignments: () => Promise<any[]>;
  getAssignmentReport: () => Promise<any>;
  exportAssignments: () => Promise<void>;
  
  // Data Management
  exportAllData: () => Promise<void>;
  importData: () => Promise<void>;
  clearAllData: () => Promise<void>;
}

const DataContext = createContext<DataContextType | undefined>(undefined);

export const DataProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  
  // User Management
  const saveUserData = async (userData: any) => {
    try {
      await AsyncStorage.setItem('userData', JSON.stringify(userData));
    } catch (error) {
      console.error('Error saving user data:', error);
    }
  };

  const getUserData = async () => {
    try {
      const userData = await AsyncStorage.getItem('userData');
      return userData ? JSON.parse(userData) : null;
    } catch (error) {
      console.error('Error getting user data:', error);
      return null;
    }
  };

  // Student Management
  const getStudents = async () => {
    try {
      const students = await AsyncStorage.getItem('students');
      return students ? JSON.parse(students) : [];
    } catch (error) {
      console.error('Error getting students:', error);
      return [];
    }
  };

  // Attendance Management
  const saveAttendance = async (record: any) => {
    try {
      const existingRecords = await AsyncStorage.getItem('attendance');
      const records = existingRecords ? JSON.parse(existingRecords) : [];
      
      // Update existing record or add new one
      const existingIndex = records.findIndex(
        (r: any) => r.rollNumber === record.rollNumber && r.date === record.date
      );
      
      if (existingIndex >= 0) {
        records[existingIndex] = record;
      } else {
        records.push(record);
      }
      
      await AsyncStorage.setItem('attendance', JSON.stringify(records));
    } catch (error) {
      console.error('Error saving attendance:', error);
    }
  };

  const getAttendanceByDate = async (date: string) => {
    try {
      const attendance = await AsyncStorage.getItem('attendance');
      const records = attendance ? JSON.parse(attendance) : [];
      return records.filter((record: any) => record.date === date);
    } catch (error) {
      console.error('Error getting attendance by date:', error);
      return [];
    }
  };

  const getTodayAttendance = async () => {
    const today = new Date().toISOString().split('T')[0];
    return getAttendanceByDate(today);
  };

  const getAttendanceReport = async (startDate: string, endDate: string) => {
    try {
      const attendance = await AsyncStorage.getItem('attendance');
      const records = attendance ? JSON.parse(attendance) : [];
      
      const filteredRecords = records.filter((record: any) => 
        record.date >= startDate && record.date <= endDate
      );

      const totalDays = new Set(filteredRecords.map((r: any) => r.date)).size;
      const totalStudents = new Set(filteredRecords.map((r: any) => r.rollNumber)).size;
      const presentCount = filteredRecords.filter((r: any) => r.status === 'present').length;
      const avgPresent = totalDays > 0 ? Math.round((presentCount / (totalDays * totalStudents)) * 100) : 0;

      return {
        totalDays,
        totalStudents,
        avgPresent,
        lastExport: 'Today'
      };
    } catch (error) {
      console.error('Error getting attendance report:', error);
      return {};
    }
  };

  const exportAttendance = async (startDate?: string, endDate?: string, format = 'whatsapp') => {
    try {
      const attendance = await AsyncStorage.getItem('attendance');
      const records = attendance ? JSON.parse(attendance) : [];
      
      let filteredRecords = records;
      if (startDate && endDate) {
        filteredRecords = records.filter((record: any) => 
          record.date >= startDate && record.date <= endDate
        );
      }

      if (format === 'whatsapp') {
        const whatsappText = generateWhatsAppAttendanceText(filteredRecords);
        await Sharing.shareAsync('data:text/plain;base64,' + btoa(unescape(encodeURIComponent(whatsappText))));
      }
      // Add PDF and CSV export logic here
    } catch (error) {
      console.error('Error exporting attendance:', error);
    }
  };

  // Assignment Management
  const getAssignments = async () => {
    try {
      const assignments = await AsyncStorage.getItem('assignments');
      return assignments ? JSON.parse(assignments) : [];
    } catch (error) {
      console.error('Error getting assignments:', error);
      return [];
    }
  };

  const saveAssignment = async (assignment: any) => {
    try {
      const existingAssignments = await AsyncStorage.getItem('assignments');
      const assignments = existingAssignments ? JSON.parse(existingAssignments) : [];
      
      const existingIndex = assignments.findIndex((a: any) => a.id === assignment.id);
      if (existingIndex >= 0) {
        assignments[existingIndex] = assignment;
      } else {
        assignments.push(assignment);
      }
      
      await AsyncStorage.setItem('assignments', JSON.stringify(assignments));
    } catch (error) {
      console.error('Error saving assignment:', error);
    }
  };

  const deleteAssignment = async (id: string) => {
    try {
      const existingAssignments = await AsyncStorage.getItem('assignments');
      const assignments = existingAssignments ? JSON.parse(existingAssignments) : [];
      const filteredAssignments = assignments.filter((a: any) => a.id !== id);
      await AsyncStorage.setItem('assignments', JSON.stringify(filteredAssignments));
    } catch (error) {
      console.error('Error deleting assignment:', error);
    }
  };

  const getUpcomingAssignments = async () => {
    try {
      const assignments = await getAssignments();
      const today = new Date().toISOString().split('T')[0];
      return assignments
        .filter((assignment: any) => assignment.dueDate >= today)
        .sort((a: any, b: any) => a.dueDate.localeCompare(b.dueDate));
    } catch (error) {
      console.error('Error getting upcoming assignments:', error);
      return [];
    }
  };

  const getAssignmentReport = async () => {
    try {
      const assignments = await getAssignments();
      const today = new Date().toISOString().split('T')[0];
      
      const total = assignments.length;
      const upcoming = assignments.filter((a: any) => a.dueDate >= today).length;
      const overdue = assignments.filter((a: any) => a.dueDate < today).length;
      
      return { total, upcoming, overdue, recent: 0 };
    } catch (error) {
      console.error('Error getting assignment report:', error);
      return {};
    }
  };

  const exportAssignments = async () => {
    try {
      const assignments = await getAssignments();
      const whatsappText = generateWhatsAppAssignmentText(assignments);
      await Sharing.shareAsync('data:text/plain;base64,' + btoa(unescape(encodeURIComponent(whatsappText))));
    } catch (error) {
      console.error('Error exporting assignments:', error);
    }
  };

  // Data Management
  const exportAllData = async () => {
    try {
      const userData = await getUserData();
      const attendance = await AsyncStorage.getItem('attendance');
      const assignments = await AsyncStorage.getItem('assignments');
      
      const exportData = {
        userData,
        attendance: attendance ? JSON.parse(attendance) : [],
        assignments: assignments ? JSON.parse(assignments) : [],
        exportDate: new Date().toISOString(),
      };

      const jsonString = JSON.stringify(exportData, null, 2);
      const fileName = `class-rep-ai-backup-${new Date().toISOString().split('T')[0]}.json`;
      
      const fileUri = FileSystem.documentDirectory + fileName;
      await FileSystem.writeAsStringAsync(fileUri, jsonString);
      await Sharing.shareAsync(fileUri);
    } catch (error) {
      console.error('Error exporting all data:', error);
    }
  };

  const importData = async () => {
    // This would implement file picker and import logic
    console.log('Import data functionality would be implemented here');
  };

  const clearAllData = async () => {
    try {
      await AsyncStorage.clear();
    } catch (error) {
      console.error('Error clearing data:', error);
    }
  };

  // Helper functions
  const generateWhatsAppAttendanceText = (records: any[]) => {
    const groupedByDate = records.reduce((acc: any, record) => {
      if (!acc[record.date]) acc[record.date] = { present: [], absent: [], exception: [] };
      acc[record.date][record.status].push(record.rollNumber);
      return acc;
    }, {});

    let text = '';
    Object.keys(groupedByDate).forEach(date => {
      const data = groupedByDate[date];
      text += `📅 Attendance Log – ${new Date(date).toLocaleDateString()}\n`;
      text += `✅ Present: ${data.present.join(', ')}\n`;
      text += `❌ Absent: ${data.absent.join(', ')}\n`;
      if (data.exception.length > 0) {
        text += `⚠️ Exception: ${data.exception.join(', ')}\n`;
      }
      text += '\n';
    });

    return text;
  };

  const generateWhatsAppAssignmentText = (assignments: any[]) => {
    let text = '📝 Assignments Due:\n';
    assignments.forEach(assignment => {
      text += `📘 ${assignment.subject} – ${assignment.requirement}\n`;
      text += `📅 Due: ${assignment.dueDate}\n`;
      if (assignment.uploadType) {
        text += `📎 ${assignment.uploadType}`;
        if (assignment.uploadLink) {
          text += `: ${assignment.uploadLink}`;
        }
        text += '\n';
      }
      text += '\n';
    });
    return text;
  };

  const value: DataContextType = {
    saveUserData,
    getUserData,
    getStudents,
    saveAttendance,
    getAttendanceByDate,
    getTodayAttendance,
    getAttendanceReport,
    exportAttendance,
    getAssignments,
    saveAssignment,
    deleteAssignment,
    getUpcomingAssignments,
    getAssignmentReport,
    exportAssignments,
    exportAllData,
    importData,
    clearAllData,
  };

  return (
    <DataContext.Provider value={value}>
      {children}
    </DataContext.Provider>
  );
};

export const useData = () => {
  const context = useContext(DataContext);
  if (context === undefined) {
    throw new Error('useData must be used within a DataProvider');
  }
  return context;
};