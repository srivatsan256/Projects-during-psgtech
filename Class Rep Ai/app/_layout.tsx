import { useEffect, useState } from 'react';
import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { useFrameworkReady } from '@/hooks/useFrameworkReady';
import { ThemeProvider } from '@/context/ThemeContext';
import { DataProvider } from '@/context/DataContext';
import SimpleOTAChecker from '@/components/SimpleOTAChecker';
import UserAuth from '@/components/UserAuth';

export default function RootLayout() {
  const [currentUser, setCurrentUser] = useState<any>(null);
  
  useFrameworkReady();

  const handleAuthSuccess = (user: any) => {
    setCurrentUser(user);
  };

  return (
    <ThemeProvider>
      <DataProvider>
        <SimpleOTAChecker>
          <UserAuth onAuthSuccess={handleAuthSuccess}>
            <Stack screenOptions={{ headerShown: false }}>
              <Stack.Screen name="login" />
              <Stack.Screen name="(tabs)" />
              <Stack.Screen name="+not-found" />
            </Stack>
            <StatusBar style="auto" />
          </UserAuth>
        </SimpleOTAChecker>
      </DataProvider>
    </ThemeProvider>
  );
}