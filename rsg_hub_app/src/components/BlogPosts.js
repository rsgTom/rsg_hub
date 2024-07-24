import React, { useEffect, useState } from 'react';
import { View, Text, FlatList, Image, StyleSheet, TouchableOpacity } from 'react-native';
import axios from 'axios';
import { useNavigation } from '@react-navigation/native';

const BlogPosts = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigation = useNavigation();

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:5000/data/blog_posts');
        setData(response.data);
      } catch (error) {
        console.error("Error fetching data: ", error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const renderItem = ({ item }) => (
    <TouchableOpacity
      style={styles.tile}
      onPress={() => navigation.navigate('Details', { item })}
    >
      <Image source={{ uri: item.Metadata.image }} style={styles.image} />
      <Text style={styles.title}>{item.Metadata.title}</Text>
      <Text style={styles.tldr}>{item.Metadata.tldr}</Text>
    </TouchableOpacity>
  );

  if (loading) {
    return <Text>Loading...</Text>;
  }

  return (
    <View>
      <FlatList
        data={data}
        keyExtractor={item => item.uuid}
        renderItem={renderItem}
        numColumns={2} // To render tiles in a grid
      />
    </View>
  );
};

const styles = StyleSheet.create({
  tile: {
    flex: 1,
    margin: 10,
    padding: 10,
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 10,
    alignItems: 'center',
  },
  image: {
    width: 100,
    height: 100,
    marginBottom: 10,
  },
  title: {
    fontWeight: 'bold',
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 5,
  },
  tldr: {
    fontSize: 14,
    textAlign: 'center',
  },
});

export default BlogPosts;
