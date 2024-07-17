import React from 'react';
import { View, Text, ScrollView, StyleSheet } from 'react-native';

const BlogPostDetails = ({ route }) => {
  const { item } = route.params;

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>{item.Metadata.title}</Text>
      <Text style={styles.keyFindingsTitle}>Key Findings:</Text>
      <Text>{item.Research.key_findings.join('\n')}</Text>
      {/* Render additional details as needed */}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  keyFindingsTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginTop: 20,
    marginBottom: 10,
  },
});

export default BlogPostDetails;
