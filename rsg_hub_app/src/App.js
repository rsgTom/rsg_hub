import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import BlogPosts from './components/BlogPosts';
import BlogPostDetails from './components/BlogPostDetails';

const Stack = createStackNavigator();

const App = () => {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Home">
        <Stack.Screen name="Home" component={BlogPosts} />
        <Stack.Screen name="Details" component={BlogPostDetails} />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default App;
