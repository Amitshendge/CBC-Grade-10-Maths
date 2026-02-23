import pymongo
import os
import dotenv
from bson import json_util
import json

# Load environment variables from .env file
dotenv.load_dotenv()

MONGO_URL = os.getenv("mongo_url")
MONGO_DB = os.getenv("mongo_db")
MONGO_CHAT_COLLECTION = os.getenv("mongo_chat_collection")
MONGO_MAPPING_COLLECTION = os.getenv("mongo_mapping_docs_collection")
MONGO_FORMS_COLLECTION = os.getenv("mongo_forms_collection")
MONGO_INPROCESS_FORMS_COLLECTION = os.getenv("mongo_inprocess_forms_collection")

class MongoUtil:
    def __init__(self, uri: str=MONGO_URL, db_name: str=MONGO_DB, chat_collection_name: str=MONGO_CHAT_COLLECTION,
                 mapping_collection_name: str=MONGO_MAPPING_COLLECTION, forms_collection_name: str=MONGO_FORMS_COLLECTION,
                 inprocess_forms_collection_name: str=MONGO_INPROCESS_FORMS_COLLECTION):
        self.client = pymongo.MongoClient(uri)
        self.db = self.client[db_name]
        self.chat_collection = self.db[chat_collection_name]
        self.mapping_collection = self.db[mapping_collection_name]
        self.forms_collection = self.db[forms_collection_name]
        self.inprocess_forms_collection = self.db[inprocess_forms_collection_name]

    def get_collection(self, collection_name: str):
        return self.db[collection_name]

    def insert_document(self, document: dict, collection = None):
        self.chat_collection.replace_one({"_id": document["_id"]}, document, upsert=True) if collection is None else collection.replace_one({"_id": document["_id"]}, document, upsert=True)
        return document["_id"]
    
    def set_document(self, document_id, document: dict, collection = None):
        """
        Update a document in the collection by its ID.
        :param document_id: The ID of the document to update.
        :param document: The updated document data.
        """
        self.chat_collection.update_one({"_id": document_id}, {"$set": document}) if collection is None else collection.update_one({"_id": document_id}, {"$set": document})
    
    # Method to find documents in the collection based on id
    def find_document_by_id(self, document_id, collection=None):
        return self.chat_collection.find_one({"_id": document_id}) if collection is None else collection.find_one({"_id": document_id})
    
    def close(self):
        self.client.close()
    
    def get_next_document(self, current_id):
        """
        Get the next document in the sequence based on the current document ID.
        :param current_id: The ID of the current document.
        :return: The next document or None if there is no next document.
        """
        current_document = self.find_document_by_id(current_id)
        next_id = current_document.get("next_question_id")
        return self.find_document_by_id(next_id) if next_id else {}
    
    def get_previous_document(self, current_id):
        """
        Get the previous document in the sequence based on the current document ID.
        :param current_id: The ID of the current document.
        :return: The previous document or None if there is no previous document.
        """
        current_document = self.find_document_by_id(current_id)
        previous_id = current_document.get("previous_question_id")
        return self.find_document_by_id(previous_id) if previous_id else {}

    def export_all_documents_to_json(self, file_path, collection=None):
        """
        Export all documents from the MongoDB collection to a JSON file.

        :param file_path: The full path to the output JSON file.
        """
        cursor = self.chat_collection.find() if collection is None else collection.find()
        
        # Convert cursor to list and serialize with json_util to handle ObjectId and other BSON types
        documents = list(cursor)
        with open(file_path, 'w') as json_file:
            json.dump(documents, json_file, default=json_util.default, indent=4)

    def drop_collection(self, collection_list=None):
        """
        Drop the entire collection.
        """
        for collection in collection_list if collection_list else [self.chat_collection, self.mapping_collection, self.forms_collection]:
            collection.drop()
            print(f"Collection '{collection.name}' dropped.")

    def delete_document_by_id(self, document_id):
        """
        Delete a document from the collection by its ID.
        :param document_id: The ID of the document to delete.
        """
        result = self.chat_collection.delete_one({"_id": document_id})
        if result.deleted_count > 0:
            print(f"Document with ID {document_id} deleted.")
        else:
            print(f"No document found with ID {document_id}.")


if __name__ == "__main__":
    # Example usage
    mongo_util = MongoUtil()
    # Insert a sample document
    # sample_document = {"name": "John Doe", "age": 30, "city": "New York"}
    # inserted_id = mongo_util.insert_document(sample_document)
    # print(f"Inserted document with ID: {inserted_id}")
    # Access a specific collection
    # print(mongo_util.find_document_by_id(inserted_id))
    # mongo_util.drop_collection()
    # abc = mongo_util.create_serialized_quetsions(sample_document)
    # print(mongo_util.get_seralized_questions(ObjectId("684c737c609e4dc99965a12a"), reverse=True))
    # for i in range(len(sample_document)):
    #     doc_stored = mongo_util.find_document_by_id(abc['_id'])
    #     answer = input(doc_stored['question'] + " : ")
    #     doc_stored['answer'] = answer
    #     mongo_util.set_document(doc_stored['_id'], doc_stored)
    #     abc['_id'] = doc_stored['next_question_id']
    mongo_util.export_all_documents_to_json('DB_data/output.json')
    mongo_util.close()

