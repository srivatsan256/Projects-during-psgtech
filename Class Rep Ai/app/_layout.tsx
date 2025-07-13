import { useEffect, useState } from 'react';
import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { useFrameworkReady } from '@/hooks/useFrameworkReady';
import { ThemeProvider } from '@/context/ThemeContext';
import { DataProvider } from '@/context/DataContext';
import SimpleOTAChecker from '@/components/SimpleOTAChecker';

export default function RootLayout() {
  useFrameworkReady();

  return (
    <ThemeProvider>
      <DataProvider>
        <SimpleOTAChecker>
          <Stack screenOptions={{ headerShown: false }}>
            <Stack.Screen name="login" />
            <Stack.Screen name="(tabs)" />
            <Stack.Screen name="+not-found" />
          </Stack>
          <StatusBar style="auto" />
        </SimpleOTAChecker>
      </DataProvider>
    </ThemeProvider>
  );
}