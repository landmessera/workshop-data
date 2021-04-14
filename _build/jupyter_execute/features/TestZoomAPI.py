token = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJhdWQiOm51bGwsImlzcyI6IlRob3ljQ3lxUi0tSlJtWFlwMGhYUFEiLCJleHAiOjE2MTgzMDY1MTQsImlhdCI6MTYxODMwMTExNH0.rV8J2WjFmBAR_IGBfXyYWqMykGYke5Nh43xCVtDIR0s'

import http.client

conn = http.client.HTTPSConnection("api.zoom.us")

headers = {
    'authorization': "Bearer "+token,
    'content-type': "application/json"
    }

conn.request("GET", "/v2/users?status=active&page_size=30&page_number=1", headers=headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))

import http.client

conn = http.client.HTTPSConnection("api.zoom.us")

headers = {
    'authorization': "Bearer "+token,
    'content-type': "application/json"
    }

conn.request("GET", "v2/metrics/meetings?from=2021-02-20&to=2021-03-10", headers=headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))

v2/metrics/meetings